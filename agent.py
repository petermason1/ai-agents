# === agent.py ===
import os
import sys
import logging
import warnings
from datetime import datetime
from urllib.parse import urlparse
import re
from dotenv import load_dotenv
from github import Github
from github.GithubException import UnknownObjectException
from slugify import slugify
import schedule
import time
import random

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun

# === Setup ===
warnings.filterwarnings('ignore', category=DeprecationWarning)
load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')
BASE_URL = os.getenv('BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/')
MODEL_ID = os.getenv('MODEL_ID', 'gemini-2.5-flash-preview-04-17')

if not API_KEY:
    print("Error: GOOGLE_API_KEY not set.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === SYSTEM PROMPT ===
system_prompt = (
    "You are an AI writing assistant. Always begin by reviewing your blog post title. "
    "Make sure it is relevant, attention-grabbing, and **under 90 characters**. "
    "Your output should be clean, direct, and suitable for posting on a smart home blog. "
    "Include a short intro, clear sections, and useful takeaways."
)

llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    model_name=MODEL_ID,
    temperature=0.7,
    max_tokens=1200,
    system_message=system_prompt
)


# === Utility Functions ===
def extract_title_description(content: str):
    paragraphs = content.strip().split('\n')
    raw_title = paragraphs[0].strip() if paragraphs else ""
    title = re.sub(r'[^\w\s-]', '', raw_title).strip()

    if not title or len(title.split()) < 3:
        title = f"Smart Home Tips {datetime.utcnow().strftime('%Y%m%d')}"
        logger.warning("⚠️ Fallback title used due to weak or missing heading.")

    title = title[:90].strip()

    description = ""
    for para in paragraphs[1:]:
        para = para.strip()
        if len(para.split()) > 6:
            description = para[:160].strip()
            break
    if not description:
        description = "Tips and ideas for improving your smart home setup."

    keywords = [
        "motion", "alexa", "home assistant", "privacy", "energy", "lighting", "routine",
        "presence", "automation", "smart plug", "camera", "security", "sensor", "voice"
    ]
    content_lower = content.lower()
    tags = sorted({word for word in keywords if word in content_lower})

    return title, description, tags

def safe_blog_post(content: str) -> str:
    token = os.getenv('BLOG_PUSH_TOKEN')
    repo_name = os.getenv('BLOG_REPO')
    if repo_name and repo_name.startswith('http'):
        repo_name = urlparse(repo_name).path.strip('/')
    branch = os.getenv('BLOG_BRANCH', 'main')
    posts_dir = os.getenv('BLOG_POSTS_DIR', 'content/posts')

    if not token or not repo_name:
        return 'Error: BLOG_PUSH_TOKEN or BLOG_REPO not set.'

    title, description, tags = extract_title_description(content)
    slug = slugify(title.lower())[:60] or f"smart-home-{datetime.utcnow().strftime('%Y%m%d')}"
    filename = f"{posts_dir}/{datetime.utcnow().date().isoformat()}-{slug}.mdx"

    frontmatter = (
        f"---\n"
        f"title: \"{title}\"\n"
        f"description: \"{description}\"\n"
        f"date: \"{datetime.utcnow().isoformat()}Z\"\n"
        f"slug: \"{slug}\"\n"
        f"published: false\n"
        f"tags: {tags}\n"
        f"---\n\n"
    )

    gh = Github(token)
    repo = gh.get_repo(repo_name)
    md_content = frontmatter + content

    try:
        existing = repo.get_contents(filename, ref=branch)
        repo.update_file(filename, f"Update post: {title}", md_content, existing.sha, branch=branch)
        return f"✅ Updated {filename}"
    except UnknownObjectException:
        repo.create_file(filename, f"New post: {title}", md_content, branch=branch)
        return f"✅ Created {filename}"
    except Exception as e:
        logger.error(f"❌ Publish error: {e}")
        return f"❌ Publish error: {e}"

# === Agent Execution ===
def generate_publish(prompt: str, retries=5):
    logger.info("Generating blog post...")
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Prompting agent with: {prompt}")
            content = agent.run(prompt)
            if not content or "i am sorry" in content.lower() or "i cannot" in content.lower():
                logger.warning(f"⚠️ AI output was a refusal or empty. Retrying (Attempt {attempt})...")
                continue
            result = safe_blog_post(content)
            logger.info(result)
            print(result)
            break
        except Exception as e:
            logger.error(f"⚠️ Generation attempt {attempt} failed: {e}")

# === CLI + Scheduler ===
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', type=str)
    parser.add_argument('-s', '--schedule', action='store_true')
    args = parser.parse_args()

    trending_prompts = [
        "Top 5 smart home trends in 2025",
        "Why Home Assistant beats Alexa for privacy",
        "How to set up smart routines with motion sensors",
        "Voice-controlled lighting: pros and cons",
        "Energy-saving automation tips for beginners",
        "How to use presence detection with smart plugs",
        "Best security automations for your smart home"
    ]
    prompt = args.question or random.choice(trending_prompts)
    logger.info(f"📢 Prompt used: {prompt}")

    def safe_calendar(action: str) -> str:
        logger.info(f"Calendar stub: {action}")
        return f"Calendar stub: {action}"

    def safe_db(query: str) -> str:
        logger.info(f"DB stub: {query}")
        return f"DB stub: {query}"

    tools = [
        Tool(name='Calendar', func=safe_calendar, description='Calendar stub'),
        Tool(name='Database', func=safe_db, description='Database stub'),
        Tool(name='BlogPost', func=safe_blog_post, description='Publish blog to GitHub')
    ]

    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model_name=MODEL_ID,
        temperature=0.7
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent='zero-shot-react-description',
        verbose=False,
        handle_parsing_errors=True
    )

    if args.schedule:
        post_time = os.getenv('POST_TIME', '08:00')
        schedule.every().day.at(post_time).do(lambda: generate_publish(prompt))
        logger.info(f"Scheduled to run daily at {post_time}")
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        generate_publish(prompt)
