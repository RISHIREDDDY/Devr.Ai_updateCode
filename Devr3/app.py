from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import requests
from dotenv import load_dotenv
import os
from flask import Flask
from supabase import create_client
import openai

# Load environment variables
load_dotenv()

# Initialize embeddings model
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    exit(1)

# Supabase setup
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)
try:
    response = supabase.table("contributors").select("*").limit(1).execute()
    print("✅ Supabase connection successful:", response)
except Exception as e:
    print("🚨 Supabase connection failed:", e)
    exit(1)

# Groq API setup
openai.api_key = os.getenv('GROQ_API_KEY')


def create_vector_store(documents):
    """Create FAISS vector store from given documents."""
    try:
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        print(f"⚠️ Error initializing FAISS vector store: {e}")
        return None


# Fetch GitHub users (contributors or reviewers)
def fetch_github_users(user_type):
    """Fetch contributors or reviewers from GitHub API based on user_type."""
    github_token = os.getenv('GITHUB_TOKEN')
    repo_owner = os.getenv('GITHUB_REPO_OWNER')
    repo_name = os.getenv('GITHUB_REPO_NAME')

    if not github_token or not repo_owner or not repo_name:
        print(f"🚨 Missing GitHub credentials! Check your .env file.")
        return []

    if user_type == "contributors":
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    elif user_type == "reviewers":
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/collaborators"
    else:
        print("🚨 Invalid user type! Use 'contributors' or 'reviewers'.")
        return []

    headers = {'Authorization': f'token {github_token}', 'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return [{"login": user["login"], "bio": user.get("bio", ""), "role": user_type} for user in response.json()]
    else:
        print(f"⚠️ Failed to fetch {user_type}: {response.status_code} - {response.text}")
        return []


# Extract expertise tags from GitHub bio
def extract_expertise_tags(bio):
    if not bio:
        return []
    words = bio.lower().split()
    tags = [f"#{word}" for word in words if len(word) > 2]
    return tags


# Generate expertise tags from PR title and body
def get_pr_insights(pr_title, pr_body):
    words = (pr_title + " " + pr_body).lower().split()
    return list(set([f"#{word}" for word in words if len(word) > 2]))


# Prepare documents for FAISS with expertise tags
def prepare_documents(users):
    documents = []
    for user in users:
        name = user['login']
        role = user['role']
        expertise_tags = extract_expertise_tags(user.get('bio', ""))
        tags = [f"#{name}", f"#{role}"] + expertise_tags

        documents.append(Document(
            page_content=" ".join(tags),
            metadata={"username": name, "role": role, "expertise": expertise_tags}
        ))
    return documents


# Store PR details in Supabase
def store_pr_data(pr_title, pr_body, pr_link, contributor, reviewer):
    data = {"pr_title": pr_title, "pr_body": pr_body, "pr_link": pr_link, "contributor": contributor,
            "reviewer": reviewer}
    supabase.table('pull_requests').insert(data).execute()


# Notify Discord
def notify_discord(user, message):
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not discord_webhook_url:
        print("🚨 Missing Discord webhook URL! Check your .env file.")
        return

    payload = {"content": f"@{user}, {message}"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(discord_webhook_url, json=payload, headers=headers)

    if response.status_code == 204:
        print(f"✅ Notification sent to Discord: {user}")
    else:
        print(f"⚠️ Failed to send Discord notification: {response.status_code} - {response.text}")


# Notify GitHub
def notify_github(pr_number, reviewer):
    github_token = os.getenv('GITHUB_TOKEN')
    repo_owner = os.getenv('GITHUB_REPO_OWNER')
    repo_name = os.getenv('GITHUB_REPO_NAME')

    if not github_token or not repo_owner or not repo_name:
        print(f"🚨 Missing GitHub credentials! Check your .env file.")
        return

    headers = {'Authorization': f'token {github_token}', 'Accept': 'application/vnd.github.v3+json'}
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/requested_reviewers"

    payload = {"reviewers": [reviewer]}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code in [201, 200]:
        print(f"✅ Review request sent to GitHub: {reviewer}")
    else:
        print(f"⚠️ Failed to send GitHub review request: {response.status_code} - {response.text}")


# Main execution
users = fetch_github_users("contributors") + fetch_github_users("reviewers")
if users:
    user_docs = prepare_documents(users)
    vector_store = create_vector_store(user_docs)
    print("✅ Vector store successfully created.")
else:
    print("⚠️ Skipping vector store creation due to missing contributors or reviewers.")
    vector_store = None

if __name__ == '__main__':
    app = Flask(__name__)
    app.run(port=5000)
