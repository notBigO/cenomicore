import asyncio
import json
import os
import sys
import argparse
from getpass import getpass
import requests
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print
from rich.live import Live
from rich.text import Text
import aiohttp
from auth import dummy_token  # Import the dummy token from auth.py

# Terminal colors and styling
console = Console()

API_URL = "http://localhost:8000"


class ChatTerminal:
    def __init__(self):
        self.conversation_id = None
        self.user_id = None
        self.mall_id = None
        self.language = "en"
        self.malls = []
        self.headers = {"Authorization": f"Bearer {dummy_token}"}  # Set up headers with the token
        self.use_streaming = True  # Default to using streaming response

    async def login(self):
        email = Prompt.ask("[bold blue]Email[/bold blue]")
        password = getpass("Password: ")

        try:
            response = requests.post(
                f"{API_URL}/login", 
                json={"email": email, "password": password},
                headers=self.headers  # Add headers with authentication token
            )
            if response.status_code == 200:
                self.user_id = response.json()["user_id"]
                console.print(
                    f"[green]Logged in successfully as {self.user_id}[/green]"
                )
                return True
            else:
                console.print("[red]Login failed. Please check your credentials.[/red]")
                return False
        except Exception as e:
            console.print(f"[red]Error during login: {str(e)}[/red]")
            return False

    async def get_malls(self):
        try:
            response = requests.get(
                f"{API_URL}/malls",
                headers=self.headers  # Add headers with authentication token
            )
            if response.status_code == 200:
                self.malls = response.json()
                console.print("[green]Available malls:[/green]")
                for i, mall in enumerate(self.malls, 1):
                    console.print(
                        f"[cyan]{i}.[/cyan] {mall['name_en']} (ID: {mall['mall_id']})"
                    )
                return True
            else:
                console.print("[red]Failed to fetch malls.[/red]")
                return False
        except Exception as e:
            console.print(f"[red]Error fetching malls: {str(e)}[/red]")
            return False

    async def select_mall(self):
        if not self.malls:
            await self.get_malls()

        if not self.malls:
            return False

        choice = Prompt.ask(
            "[bold blue]Select a mall (number)[/bold blue]",
            choices=[str(i) for i in range(1, len(self.malls) + 1)],
        )

        selected_index = int(choice) - 1
        self.mall_id = int(self.malls[selected_index]["mall_id"])
        mall_name = self.malls[selected_index]["name_en"]
        console.print(f"[green]Selected mall: {mall_name} (ID: {self.mall_id})[/green]")
        return True

    async def send_message(self, text):
        if not self.mall_id:
            console.print("[yellow]Please select a mall first![/yellow]")
            await self.select_mall()

        if self.use_streaming:
            return await self.send_streaming_message(text)
        else:
            return await self.send_regular_message(text)

    async def send_regular_message(self, text):
        try:
            payload = {
                "text": text,
                "conversation_id": self.conversation_id,
                "language": self.language,
                "mall_id": self.mall_id,
            }

            if self.user_id:
                payload["user_id"] = self.user_id

            response = requests.post(
                f"{API_URL}/chat", 
                json=payload,
                headers=self.headers  # Add headers with authentication token
            )

            if response.status_code == 200:
                data = response.json()
                self.conversation_id = data["conversation_id"]
                return data["message"]
            else:
                # Log the error but don't display technical details to the user
                console.print(f"[red]Error: {response.status_code}[/red]", style="dim")

                # Only log the response text, don't show it to the user
                if hasattr(console, "log"):
                    console.log(f"API Error response: {response.text}")

                return "I'm sorry, I'm having trouble understanding that right now. Could you try rephrasing your question?"
        except Exception as e:
            # Log the error but don't display it to the user
            console.print(f"[red]Error sending message[/red]", style="dim")
            if hasattr(console, "log"):
                console.log(f"Exception: {str(e)}")

            return "I apologize, but I'm having technical difficulties right now. Please try again later."

    async def send_streaming_message(self, text):
        try:
            payload = {
                "text": text,
                "conversation_id": self.conversation_id,
                "language": self.language,
                "mall_id": self.mall_id,
            }

            if self.user_id:
                payload["user_id"] = self.user_id

            full_response = ""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_URL}/chat/stream", 
                    json=payload,
                    headers=self.headers
                ) as resp:
                    if resp.status != 200:
                        console.print(f"[red]Error: {resp.status}[/red]", style="dim")
                        return "I'm sorry, I'm having trouble understanding that right now. Could you try rephrasing your question?"
                    
                    # Create a live display for the streaming response
                    console.print(f"[bold blue]Assistant: [bright_green](streaming)[/bright_green][/bold blue]")
                    
                    # Use live display with a counter to show streaming progress
                    stream_count = 0
                    with Live(
                        Panel(Markdown(full_response), border_style="blue"),
                        refresh_per_second=20  # Higher refresh rate
                    ) as live:
                        async for line in resp.content:
                            if not line.strip():
                                continue
                                
                            try:
                                chunk_data = json.loads(line)
                                
                                if chunk_data.get("type") == "start":
                                    self.conversation_id = chunk_data.get("conversation_id")
                                    live.update(Panel(
                                        Markdown(""),
                                        border_style="blue",
                                        subtitle="[dim]Starting stream...[/dim]"
                                    ))
                                    
                                elif chunk_data.get("type") == "chunk":
                                    chunk_text = chunk_data.get("content", "")
                                    full_response += chunk_text
                                    stream_count += 1
                                    
                                    # Add a visual indicator of streaming by updating the subtitle
                                    indicator = "🔄 " + "·" * (stream_count % 4 + 1)
                                    
                                    # Update the live display with the current full response
                                    live.update(Panel(
                                        Markdown(full_response),
                                        border_style="blue",
                                        subtitle=f"[dim]{indicator} Receiving stream ({stream_count} chunks)[/dim]"
                                    ))
                                    
                                elif chunk_data.get("type") == "error":
                                    error_msg = chunk_data.get("message", "An error occurred")
                                    full_response = error_msg
                                    live.update(Panel(
                                        Markdown(error_msg),
                                        border_style="red",
                                        subtitle="[dim red]Stream Error[/dim red]"
                                    ))
                                
                                elif chunk_data.get("type") == "end":
                                    live.update(Panel(
                                        Markdown(full_response),
                                        border_style="blue",
                                        subtitle=f"[dim]Stream complete ({stream_count} chunks received)[/dim]"
                                    ))
                                    
                                    # Check for recommendations and follow-up questions
                                    if chunk_data.get("recommendations"):
                                        recommendations = chunk_data.get("recommendations", [])
                                        if recommendations:
                                            rec_text = "\n\n[bold cyan]Recommendations:[/bold cyan]\n"
                                            for rec in recommendations:
                                                rec_text += f"• {rec.get('title')}\n"
                                            live.update(Panel(
                                                Markdown(full_response + rec_text),
                                                border_style="blue"
                                            ))
                                    
                                    if chunk_data.get("follow_up_question"):
                                        follow_up = chunk_data.get("follow_up_question")
                                        follow_up_text = f"\n\n[bold green]Follow-up:[/bold green] {follow_up}"
                                        live.update(Panel(
                                            Markdown(full_response + follow_up_text),
                                            border_style="blue"
                                        ))
                                    
                            except json.JSONDecodeError:
                                console.print("[yellow]Warning: Received malformed JSON from streaming endpoint[/yellow]", style="dim")
                    
            if stream_count > 0:
                console.print(f"[dim]Received {stream_count} streaming chunks in total[/dim]")
            return full_response
                
        except Exception as e:
            console.print(f"[red]Error streaming message[/red]", style="dim")
            if hasattr(console, "log"):
                console.log(f"Exception: {str(e)}")
            return "I apologize, but I'm having technical difficulties right now. Please try again later."

    async def start_chat(self):
        console.print(
            Panel.fit(
                "[bold magenta]Cenomi Mall Assistant[/bold magenta]\n"
                + "[italic]Type 'exit' to quit, 'login' to authenticate, 'mall' to select a mall, or 'stream' to toggle streaming[/italic]"
            )
        )

        await self.get_malls()
        await self.select_mall()

        # Show streaming status at start
        streaming_status = "[green]enabled[/green]" if self.use_streaming else "[red]disabled[/red]"
        console.print(f"[blue]Streaming mode is {streaming_status}[/blue]")

        while True:
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if user_input.lower() == "exit":
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif user_input.lower() == "login":
                await self.login()
                continue
            elif user_input.lower() == "mall":
                await self.select_mall()
                continue
            elif user_input.lower() == "stream":
                self.use_streaming = not self.use_streaming
                streaming_status = "[green]enabled[/green]" if self.use_streaming else "[red]disabled[/red]"
                console.print(f"[blue]Streaming mode {streaming_status}[/blue]")
                console.print(f"[dim]{'Responses will be shown as they are generated.' if self.use_streaming else 'Responses will be shown after completion.'}")
                continue

            if self.use_streaming:
                # For streaming mode, we show the response during send_streaming_message
                response = await self.send_message(user_input)
            else:
                # For non-streaming mode, show "Thinking..." and then the full response
                with console.status("[bold blue]Thinking...[/bold blue]"):
                    response = await self.send_message(user_input)
                
                console.print(
                    Panel(
                        Markdown(response),
                        title="[bold blue]Assistant[/bold blue]",
                        border_style="blue",
                    )
                )


async def main():
    parser = argparse.ArgumentParser(description="Cenomi Mall Chatbot Terminal")
    parser.add_argument(
        "--api",
        help="API URL (default: http://localhost:8000)",
        default="http://localhost:8000",
    )
    args = parser.parse_args()

    global API_URL
    API_URL = args.api

    terminal = ChatTerminal()
    await terminal.start_chat()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)
