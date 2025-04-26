# Cenomi Chatbot User Guide

Welcome to the Cenomi Chatbot system! This guide will help you navigate the features and capabilities of the chatbot for both mall customers and tenant store managers.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Customer Guide](#customer-guide)
3. [Tenant Guide](#tenant-guide)
4. [Command Reference](#command-reference)
5. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements

- A computer or mobile device with a modern web browser
- Internet connection
- For terminal use: Python 3.8+ installed

### Accessing the Chatbot

There are two ways to access the Cenomi Chatbot:

1. **Terminal Interface**: Run the chat terminal application:

   ```bash
   python src/chat_terminal.py
   ```

2. **API Integration**: Developer access via HTTP requests to the API endpoints:
   ```
   http://localhost:8000/chat
   http://localhost:8000/tenant/update
   ```

### Login and Authentication

The first time you use the chatbot, you may want to log in:

1. In the terminal interface, type `login`
2. Enter your email address
3. Enter your password
4. If successful, you'll see a confirmation message with your user ID

## Customer Guide

The customer chatbot helps mall visitors find information about stores, products, offers, and facilities.

### Selecting a Mall

Before starting, select which mall you're interested in:

1. Type `mall` in the chat terminal
2. Choose a mall from the list by entering its number
3. The chatbot will confirm your selection

### Types of Customer Queries

#### Finding Stores

You can ask about store locations in various ways:

- "Where is Zara located?"
- "I'm looking for shoe stores"
- "Which floor is the food court on?"
- "Are there any electronics stores in the mall?"

The chatbot will provide store names, locations, and sometimes additional information like operating hours.

#### Product Inquiries

Ask about products available in the mall:

- "Where can I find iPhones?"
- "Which stores sell children's clothing?"
- "Do any stores have running shoes on sale?"
- "I'm looking for a gift for my mom"

#### Mall Facilities

Inquire about mall amenities and services:

- "Where are the restrooms?"
- "Is there a prayer room in this mall?"
- "Where can I find ATMs?"
- "Do you have wheelchair access?"

#### Promotions and Offers

Ask about current deals and promotions:

- "What sales are happening now?"
- "Are there any discounts at clothing stores?"
- "Tell me about current promotions"
- "Does Starbucks have any special offers?"

#### Navigation Assistance

Get help finding your way around:

- "How do I get from Zara to the food court?"
- "Where are the elevators?"
- "I'm near H&M, where's the nearest restroom?"
- "Where can I find parking?"

#### Follow-up Questions

The chatbot remembers your conversation context, so you can ask follow-up questions:

- "What floor is it on?" (after asking about a store)
- "Are there any others?" (after receiving recommendations)
- "What time do they open?" (referring to a previously mentioned store)

### Tips for Better Customer Responses

- **Be specific**: Mention store names, product types, or locations when possible
- **Provide context**: If you're looking for directions, mention where you currently are
- **Ask follow-ups**: If the initial response isn't detailed enough, ask for more information
- **Specify preferences**: Mention price ranges, styles, or other preferences for better recommendations

## Tenant Guide

Tenant store managers can use the chatbot to manage store information, products, and promotional offers.

### Authentication

Tenant accounts start with the prefix `t_` in the user ID. To access tenant features:

1. Type `login` in the terminal
2. Enter your tenant email and password
3. The system will authenticate and recognize you as a tenant

### Managing Your Store

If you have multiple stores, the chatbot will ask you to select one:

1. When prompted, enter the number corresponding to the store you want to manage
2. The chatbot will confirm your selection
3. You can switch stores at any time by mentioning a different store name

### Managing Offers

You can create, update, delete, and list promotional offers:

#### Creating an Offer

Say something like "I want to create a new offer" or "Add a promotion for my store"

The chatbot will guide you through the process, asking for:

1. Offer description (e.g., "20% off all shoes")
2. Start date (e.g., "2023-12-01" or "today")
3. End date (e.g., "2023-12-31")

Example conversation:

```
You: I want to add a new promotion
Assistant: What's the offer for Zara? (e.g., '20% off summer clothes')
You: 30% off all jeans
Assistant: When should it start? (e.g., 'today' or '2023-12-01')
You: today
Assistant: When should it end? (e.g., '2023-12-31')
You: 2023-12-31
Assistant: Added '30% off all jeans' to Zara from today to 2023-12-31. Anything else? 😊
```

#### Updating an Offer

Say something like "I need to update an offer" or "Change a promotion"

The chatbot will:

1. List your current offers
2. Ask which one to update
3. Ask what to change (description, start date, or end date)
4. Ask for the new value

#### Deleting an Offer

Say something like "Remove an offer" or "Delete a promotion"

The chatbot will:

1. List your current offers
2. Ask which one to delete
3. Confirm the deletion

#### Listing Offers

Say something like "Show my offers" or "List all promotions"

The chatbot will display all current offers for your store with their dates.

### Managing Products

You can create, update, delete, and list products:

#### Creating a Product

Say something like "Add a new product" or "Create a product listing"

The chatbot will ask for:

1. Product name
2. Description
3. Price
4. Currency

Example conversation:

```
You: I want to add a new product
Assistant: What's the product name for Zara? (e.g., 'Blue Shirt')
You: Slim Fit Jeans
Assistant: What's the description for 'Slim Fit Jeans'? (e.g., 'Cotton, size M')
You: Dark wash denim, available in sizes 28-38
Assistant: How much does 'Slim Fit Jeans' cost? (e.g., '50')
You: 199
Assistant: What's the currency for 'Slim Fit Jeans'? (e.g., 'SAR')
You: SAR
Assistant: Added 'Slim Fit Jeans' (Dark wash denim, available in sizes 28-38) to Zara for 199 SAR. Anything else? 😊
```

#### Updating a Product

Say something like "Update a product" or "Change product information"

The chatbot will:

1. List your current products
2. Ask which one to update
3. Ask what to change (name, description, price, or currency)
4. Ask for the new value

#### Deleting a Product

Say something like "Remove a product" or "Delete a product"

The chatbot will:

1. List your current products
2. Ask which one to delete
3. Confirm the deletion

#### Listing Products

Say something like "Show my products" or "List all products"

The chatbot will display all current products for your store with their details.

### Tips for Tenant Users

- **Use natural language**: You can communicate conversationally with the chatbot
- **Combine information**: Say "Add a new product called Red Shirt priced at 150 SAR" to provide multiple details at once
- **Refer to specific stores**: If you manage multiple stores, specify which one you're updating
- **Check your work**: Use the list commands to verify your changes

## Command Reference

### Terminal Commands

The chat terminal supports the following special commands:

| Command | Description                       |
| ------- | --------------------------------- |
| `exit`  | Exits the chat terminal           |
| `login` | Initiates the login process       |
| `mall`  | Starts the mall selection process |

### Query Examples

#### Customer Queries

| Query Type     | Example                               |
| -------------- | ------------------------------------- |
| Store Location | "Where is Apple Store?"               |
| Store Category | "Show me all sports stores"           |
| Product Search | "Where can I buy a laptop?"           |
| Facilities     | "Where are the baby changing rooms?"  |
| Navigation     | "How do I get to the cinema?"         |
| Offers         | "What discounts are available today?" |

#### Tenant Queries

| Query Type     | Example                                   |
| -------------- | ----------------------------------------- |
| Create Offer   | "Add a new promotion for 50% off"         |
| Update Offer   | "Change the end date of my summer sale"   |
| Delete Offer   | "Remove the holiday discount"             |
| List Offers    | "Show all my current promotions"          |
| Create Product | "Add a new product called Premium Hoodie" |
| Update Product | "Update the price of our bestseller"      |
| Delete Product | "Remove the discontinued item"            |
| List Products  | "List all products in my store"           |

## Troubleshooting

### Common Issues

#### Authentication Problems

**Issue**: "Login failed. Please check your credentials."  
**Solution**: Verify your email and password. Contact mall administration if you continue to have issues.

#### Mall Selection Issues

**Issue**: Cannot select a mall or no malls appear  
**Solution**: Check your internet connection or restart the terminal. The API service might be down.

#### Conversation Context Loss

**Issue**: The chatbot doesn't remember previous context  
**Solution**: Stay within the same session. If the issue persists, try mentioning the topic again explicitly.

#### Tenant Operation Failures

**Issue**: "Failed to add product/offer"  
**Solution**: Check that you've provided all required information and that it's in the correct format.

### Getting Help

If you encounter issues not covered in this guide:

1. For technical problems, contact the IT support team
2. For account-related issues, contact mall administration
3. For general inquiries, use the chatbot's help function by typing "help" or "I need assistance"
