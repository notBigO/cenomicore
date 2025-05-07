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

    async def login(self):
        email = Prompt.ask("[bold blue]Email[/bold blue]")
        password = getpass("Password: ")

        try:
            response = requests.post(
                f"{API_URL}/login", json={"email": email, "password": password}
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
            response = requests.get(f"{API_URL}/malls")
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

        try:
            payload = {
                "text": text,
                "conversation_id": self.conversation_id,
                "language": self.language,
                "mall_id": self.mall_id,
            }

            if self.user_id:
                payload["user_id"] = self.user_id

            response = requests.post(f"{API_URL}/chat", json=payload)

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

    async def start_chat(self):
        console.print(
            Panel.fit(
                "[bold magenta]Cenomi Mall Assistant[/bold magenta]\n"
                + "[italic]Type 'exit' to quit, 'login' to authenticate, or 'mall' to select a mall[/italic]"
            )
        )

        await self.get_malls()
        await self.select_mall()

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
