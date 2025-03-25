import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from collections import defaultdict

EPISODE_IDS = {
    "3827": 'ThePeterAttiaDrive',
    "1524727": 'HubermanLab',
    "2811": 'TheRichRollPodcast',
    "2121": 'TheTimFerrissShow',
    "2372500": 'BenGreenfieldLife',
    "2663": 'FoundMyFitness',
    "130850950": 'TheDrHymanShow',
    "838": 'MindPump',
    "2628": 'TheModelHealthShow',
    "2155503": 'Lifespan',
    "3337": 'OptimalHealthDaily',
    "302": 'TheGeniusLife',
}

EPISODE_URL = "https://app.podscribe.ai/episode/{}"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SAVE_FOLDER = "../data/raw_transcripts"

def get_episode_links():
    links = defaultdict(list)

    for episode_id, show_title in EPISODE_IDS.items():
        backend_call = f"https://backend.podscribe.ai/api/series/{episode_id}/episodes-list?numEpisodes=100000"
        response = requests.get(backend_call, headers=HEADERS)
        data = response.json()

        for episode in data['episodes']:
            links[show_title].append((episode['id'], episode['uploadedAt'].split('T')[0]))

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
        for show_title in episode_links.keys():
            print(f'Starting transcription for {show_title} podcast')
            print('-'*50)

            for i, (episode_id, date) in enumerate(episode_links[show_title], 1):
                filename = os.path.join(SAVE_FOLDER, f"{show_title}_{date}.txt")
                # Skip if file already exists
                if os.path.exists(filename):
                    print(f"Skipping {filename} (already exists)")
                    continue

                print(f'Transcribing podcast {i}/{len(episode_links[show_title])}')

                try:
                    transcript = scrape_transcript_selenium(driver, episode_id)
                    if not transcript:
                        print(f"Transcript empty for {date}")
                        continue
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                    print(f"Saved {filename}")
                except Exception as e:
                    print(f"Failed on Episode ID {episode_id}: {e}")
                    continue
            
            print('-'*50)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()