import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

EPISODES_API = "https://backend.podscribe.ai/api/series/1524727/episodes-list?numEpisodes=100000"
EPISODE_URL = "https://app.podscribe.ai/episode/{}"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SAVE_FOLDER = "./data/raw_transcripts"

def get_episode_links():
    response = requests.get(EPISODES_API, headers=HEADERS)
    data = response.json()

    links = [
        (episode['id'], episode['uploadedAt'].split('T')[0])
        for episode in data['episodes']
    ]
    return links

def scrape_transcript_selenium(driver, episode_id):
    driver.get(EPISODE_URL.format(episode_id))
    time.sleep(5)

    transcript_divs = driver.find_elements(By.TAG_NAME, "div")
    transcript = ""

    for div in transcript_divs:
        if "transcript" in div.get_attribute("class").lower() or "transcript" in div.text.lower():
            transcript = div.text.strip()
            break

    return transcript

def main():
    episode_links = get_episode_links()
    if not episode_links:
        print("No episode links found — possible anti-bot measures or API change.")
        return

    # Create folder if it doesn't exist
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        for i, (episode_id, date) in enumerate(episode_links, 1):
            print(f'Transcribing podcast {i}/{len(episode_links)}')
            try:
                transcript = scrape_transcript_selenium(driver, episode_id)
                if not transcript:
                    print(f"Transcript empty for {date}")
                    continue
                
                filename = os.path.join(SAVE_FOLDER, f"{date}.txt")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Failed on Episode ID {episode_id}: {e}")
                continue
    finally:
        driver.quit()

if __name__ == "__main__":
    main()