# === agent.py ===
import os
import sys
import logging
import warnings
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from github import Github
from slugify import slugify
import schedule
import time
import random

# === LangChain ===
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun

# === Setup ===
warnings.filterwarnings('ignore', category=DeprecationWarning)
load_dotenv()

API_KEY       = os.getenv('GOOGLE_API_KEY')
BASE_URL      = os.getenv('BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/')
MODEL_ID      = os.getenv('MODEL_ID', 'gemini-2.5-flash-preview-04-17')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful assistant. Think step-by-step and use tools only when needed.')

if not API_KEY:
    print("Error: GOOGLE_API_KEY not set.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Helpers ===
def safe_search(query: str) -> str:
    try:
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {e}"

def safe_calendar(action: str) -> str:
    logger.info(f"Calendar stub: {action}")
    return f"Calendar stub: {action}"

def safe_db(query: str) -> str:
    logger.info(f"DB stub: {query}")
    return f"DB stub: {query}"

def extract_title_description(content: str):
    lines = content.strip().split('\n')
    title = lines[0][:40].strip()
    description = ''
    for line in lines[1:]:
        if len(line.strip()) > 40:
            description = line.strip()[:140]
            break
    return title, description

def safe_blog_post(content: str) -> str:
    from base64 import b64encode

    token     = os.getenv('BLOG_PUSH_TOKEN')
    repo_name = os.getenv('BLOG_REPO')
    if repo_name and repo_name.startswith('http'):
        path = urlparse(repo_name).path
        repo_name = path.strip('/')
    branch    = os.getenv('BLOG_BRANCH', 'main')
    posts_dir = os.getenv('BLOG_POSTS_DIR', 'content/posts')

    if not token or not repo_name:
        return 'Error: BLOG_PUSH_TOKEN or BLOG_REPO not set.'

    title, description = extract_title_description(content)
    slug       = slugify(title)
    date_str   = datetime.utcnow().date().isoformat()
    filename   = f"{posts_dir}/{date_str}-{slug}.mdx"

    frontmatter = (
        f"---\n"
        f"title: \"{title}\"\n"
        f"description: \"{description}\"\n"
        f"date: \"{datetime.utcnow().isoformat()}Z\"\n"
        f"slug: \"{slug}\"\n"
        f"published: true\n"
        f"---\n\n"
    )
    md_content = frontmatter + content

    gh = Github(token)
    repo = gh.get_repo(repo_name)

    try:
        # Try to get the file first
        existing_file = repo.get_contents(filename, ref=branch)
        sha = existing_file.sha
        repo.update_file(
            path=filename,
            message=f"Update post: {title}",
            content=md_content,
            sha=sha,
            branch=branch
        )
        return f"✅ Updated {filename} in {repo_name}@{branch}."
    except Exception as e:
        if "404" in str(e):
            # File does not exist — create it
            repo.create_file(
                path=filename,
                message=f"Automated post: {title}",
                content=md_content,
                branch=branch
            )
            return f"✅ Created {filename} in {repo_name}@{branch}."
        else:
            logger.error('Publish error: %s', e)
            return f"❌ Publish error: {e}"


# === Trending Prompt Ideas ===
trending_prompts = [
    "Top 5 smart home trends in 2025",
    "Why Home Assistant beats Alexa for privacy",
    "How to set up smart routines with motion sensors",
    "Voice-controlled lighting: pros and cons",
    "Energy-saving automation tips for beginners",
    "How to use presence detection with smart plugs",
    "Best security automations for your smart home"
]

# === Main ===
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run AI smart home blog agent')
    parser.add_argument('-q', '--question', type=str, help='Prompt for the blog post')
    parser.add_argument('-s', '--schedule', action='store_true', help='Run daily at POST_TIME from .env')
    args = parser.parse_args()

    prompt = args.question or random.choice(trending_prompts)

    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model_name=MODEL_ID,
        temperature=0.7
    )

    tools = [
        Tool(name='Search', func=safe_search, description='Web search'),
        Tool(name='Calendar', func=safe_calendar, description='Calendar stub'),
        Tool(name='Database', func=safe_db, description='Database stub'),
        Tool(name='BlogPost', func=safe_blog_post, description='Publish blog to GitHub')
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent='zero-shot-react-description',
        verbose=False,
        handle_parsing_errors=True
    )

    def generate_publish():
        logger.info('Generating blog post...')
        try:
            content = agent.run(prompt)
            result = safe_blog_post(content)
            logger.info(result)
            print(result)
        except Exception as e:
            logger.error(f"Generation failed: {e}")

    if args.schedule:
        post_time = os.getenv('POST_TIME', '08:00')
        schedule.every().day.at(post_time).do(generate_publish)
        logger.info(f"Scheduled daily post at {post_time}")
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        generate_publish()
