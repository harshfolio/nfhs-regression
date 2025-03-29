import streamlit as st
import os
import json
import asyncio
import pandas as pd
import tempfile
import time
import nest_asyncio
from io import StringIO
from datetime import datetime

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# Apply nest_asyncio to allow running asyncio within Streamlit
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="1mg Scraper with Crawl4AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .success {
        color: green;
    }
    .error {
        color: red;
    }
    .info {
        color: blue;
    }
</style>
""", unsafe_allow_html=True)

# Define the data structure for medicine product information
class MedicineProduct(BaseModel):
    name: str = Field(..., description="Name of the medicine product")
    manufacturer: str = Field(..., description="Manufacturer/company name")
    price: str = Field(..., description="Price of the product including currency")
    description: str = Field(..., description="Product description or information")
    uses: list[str] = Field(default=[], description="List of uses or indications")
    ingredients: list[str] = Field(default=[], description="List of active ingredients")
    dosage_form: str = Field(default="", description="Form of the medicine (tablet, syrup, etc.)")
    prescription_required: bool = Field(default=False, description="Whether prescription is required")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'results' not in st.session_state:
    st.session_state.results = []

if 'products_df' not in st.session_state:
    st.session_state.products_df = None

# Function to display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.markdown(content)

# Function to parse URLs from input text
def parse_urls(text):
    import re
    # Match URLs with or without http/https prefix
    url_pattern = r'https?://(?:www\.)?1mg\.com/\S+|(?:www\.)?1mg\.com/\S+'
    matches = re.findall(url_pattern, text)
    
    # Ensure all URLs have http/https prefix
    processed_urls = []
    for url in matches:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        processed_urls.append(url)
    
    return processed_urls

# Custom logging class to capture logs
class StreamlitLogCapture:
    def __init__(self, progress_bar=None):
        self.log_output = StringIO()
        self.progress_bar = progress_bar
        
    def write(self, text):
        self.log_output.write(text)
        # Update the progress bar message
        if self.progress_bar:
            self.progress_bar.text(text.strip())
        return len(text)
    
    def flush(self):
        pass
    
    def get_logs(self):
        return self.log_output.getvalue()

# Function to display a download button for the DataFrame
def create_download_link(df):
    csv = df.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"1mg_products_{timestamp}.csv"
    
    return csv, filename

# Function to scrape 1mg products
async def scrape_1mg_products(urls, gemini_model, api_key, progress_bar, progress_text):
    results = []
    
    for i, url in enumerate(urls):
        progress_text.text(f"Scraping product {i+1}/{len(urls)}: {url}")
        progress_bar.progress((i) / len(urls))
        
        try:
            # Configure the browser
            browser_config = BrowserConfig(
                headless=True,
                verbose=True
            )
            
            # Create an extraction strategy using Gemini
            extraction_strategy = LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider=f"google/{gemini_model}",
                    api_token=api_key
                ),
                schema=MedicineProduct.model_json_schema(),
                extraction_type="schema",
                instruction="""
                Extract detailed information about this medicine product from 1mg.com.
                Include:
                - Full product name
                - Manufacturer/company name
                - Exact price as displayed (with currency)
                - Brief product description
                - List of primary uses or indications
                - Active ingredients (as a list)
                - Form of the medicine (tablet, syrup, capsule, etc.)
                - Whether prescription is required (true/false)
                
                Only extract information that is explicitly mentioned on the page.
                If any field is not available, leave it empty or set to default.
                """
            )
            
            # Configure the crawler run
            run_config = CrawlerRunConfig(
                extraction_strategy=extraction_strategy,
                cache_mode=CacheMode.BYPASS,  # Skip cache to get fresh data
                word_count_threshold=1,  # Process even small amounts of text
                js_code="""
                // Close any popups or modals that might interfere with scraping
                setTimeout(() => {
                  const closeButtons = document.querySelectorAll('[aria-label="Close"], .close-btn, .dismiss-btn');
                  closeButtons.forEach(btn => btn.click());
                }, 2000);
                """
            )
            
            progress_text.text(f"Starting to scrape: {url}")
            
            # Initialize and run the crawler
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=run_config
                )
                
                if result.success:
                    progress_text.text(f"Successfully scraped {url}")
                    try:
                        parsed_content = json.loads(result.extracted_content)
                        results.append(parsed_content)
                    except json.JSONDecodeError:
                        progress_text.text(f"Error parsing JSON from {url}")
                else:
                    progress_text.text(f"Failed to scrape {url}: {result.error_message}")
        
        except Exception as e:
            progress_text.text(f"Error scraping {url}: {str(e)}")
    
    progress_bar.progress(1.0)
    progress_text.text("Scraping completed!")
    
    return results

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    gemini_model = st.selectbox(
        "Select Gemini Model",
        ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses Crawl4AI to scrape product information from 1mg.com.
    
    Enter URLs or a description of what you want to scrape in the chat.
    
    The app will extract structured data and provide a downloadable CSV.
    """)

# Main content
st.title("1mg Scraper with Crawl4AI 🔍")
st.markdown("Chat with the scraper to extract product information from 1mg.com")

# Display chat history
display_chat_messages()

# Display product data if available
if st.session_state.products_df is not None:
    st.markdown("## Scraped Products")
    st.dataframe(st.session_state.products_df)
    
    # Create download button
    csv_data, filename = create_download_link(st.session_state.products_df)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )

# Chat input
if prompt := st.chat_input("Enter URLs or describe what you want to scrape"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the new message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if API key is provided
    if not gemini_api_key:
        with st.chat_message("assistant"):
            st.markdown("⚠️ Please enter your Gemini API key in the sidebar to continue.")
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please enter your Gemini API key in the sidebar to continue."})
    else:
        # Extract URLs from input
        urls = parse_urls(prompt)
        
        # Respond based on input
        with st.chat_message("assistant"):
            if urls:
                st.markdown(f"Found {len(urls)} 1mg URLs to scrape:")
                for url in urls:
                    st.markdown(f"- {url}")
                
                st.markdown("Starting to scrape these products...")
                
                # Initialize progress bar
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Set up event loop for async operation
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(
                    scrape_1mg_products(urls, gemini_model, gemini_api_key, progress_bar, progress_text)
                )
                
                if results:
                    st.session_state.results = results
                    st.session_state.products_df = pd.DataFrame(results)
                    
                    st.markdown("### Scraping Results")
                    st.dataframe(st.session_state.products_df)
                    
                    # Create download button
                    csv_data, filename = create_download_link(st.session_state.products_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
                    response_content = f"✅ Successfully scraped {len(results)} products! You can download the CSV using the button above."
                else:
                    response_content = "❌ No products were successfully scraped. Please check the URLs and try again."
            else:
                # No URLs found, provide guidance
                response_content = """
                I didn't find any 1mg.com URLs in your message. 
                
                You can:
                1. Provide specific 1mg.com product URLs to scrape
                2. Ask me to explain how to use this app
                
                Example URLs:
                - https://www.1mg.com/drugs/crocin-advance-tablet-138496
                - https://www.1mg.com/drugs/dolo-650-tablet-150385
                """
            
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

# Add instructions at the bottom
with st.expander("How to use this app"):
    st.markdown("""
    ## Instructions
    
    1. Enter your Gemini API key in the sidebar
    2. Select the Gemini model you want to use
    3. Enter 1mg.com product URLs in the chat input
    4. The app will scrape the products and display the results
    5. Download the CSV file with the extracted data
    
    ## Example URLs
    
    ```
    https://www.1mg.com/drugs/crocin-advance-tablet-138496
    https://www.1mg.com/drugs/dolo-650-tablet-150385
    https://www.1mg.com/drugs/azithral-500-tablet-14367
    ```
    
    ## Troubleshooting
    
    If you encounter issues:
    
    - Make sure your API key is correct
    - Check if the URLs are valid 1mg.com product pages
    - Try a different Gemini model
    """)
