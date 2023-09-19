from wordcloud import WordCloud
import matplotlib.pyplot as plt
import assemblyai as aai

aai.settings.api_key = f"da736a99e91349a08bba39606ed84450"


file_urls = ["https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav", "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0011_8k.wav"]

transcriber = aai.Transcriber()
transcripts = []


for file_url in file_urls:
    transcript = transcriber.transcribe(file_url)
    transcripts.append(transcript.text)


res = " ".join(transcripts)



wcd = WordCloud(width=1200 , height=800 , background_color = 'white').generate(res)
wcd.to_file('./wordclouds/result.png')

plt.figure(figsize = (10 , 5))
plt.imshow(wcd, interpolation="bilinear")
plt.axis("off")
plt.show()