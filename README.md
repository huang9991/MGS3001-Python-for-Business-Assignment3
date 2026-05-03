# MGS3001-Python-for-Business-Assignment3
# AI-Enabled Democratization of Psychoeducation: Evidence from Reddit

**Author:** Kexin Huang | MGS 3001 WHS01

## Research Question
Does the public release of ChatGPT (November 30, 2022) correspond with 
discontinuous changes in psychoeducational content, psychological terminology 
use, and emotional supportiveness in Reddit mental health communities?

## Dataset
- Source: Arctic Shift Reddit Archive (https://arctic-shift.photon-reddit.com)
- Subreddits: r/mentalhealth, r/anxiety, r/depression, r/psychology
- Period: September 2021 – April 2026
- Format: JSONL → processed CSV
- Size: ~1000,000+ posts and comments

## How to Run
1. pip install pandas numpy scipy statsmodels scikit-learn textblob nltk tqdm
2. python -c "import nltk; nltk.download('vader_lexicon')"
3. Place JSONL files in ./data/ following naming: r_{subreddit}_posts.jsonl
4. Run: Just execute the data_collection.ipynb file sequentially/cell by cell
<img width="468" height="576" alt="image" src="https://github.com/user-attachments/assets/3c218345-156f-4a91-8f1c-7e3e8bcca777" />
