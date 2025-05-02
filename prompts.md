# All AI-Tool Prompts

## Step 1 Prompts
- Write the pip install commands needed for:
beautifulsoup4, langchain, sentence-transformers, numpy, faiss-cpu, transformers, torch

- Generate a requirements.txt listing exactly those seven libraries (one per line).

## Step 2 Prompts
- Write a Python function called scrape_webpage(url) that uses requests to fetch https://en.wikipedia.org/wiki/Rage_Against_the_Machine, parses it with BeautifulSoup, extracts all &lt;p&gt; tags inside &lt;div class='mw-parser-output'&gt;, joins their text with blank lines, writes the result to Selected_Document.txt (UTF‑8), prints a success/failure message based on the HTTP status code, and returns the article text. Please hard‑code the URL in the function. Also include a main() function and an if __name__ == '__main__': block that calls scrape_webpage() so the script runs when executed.

### Debugging prompts
- It's making the document but there's no text in it

- ⚠️ No &lt;p&gt; tags found inside the content container.

 - No &lt;p&gt; tags found inside the content container.
&lt;div class="mw-parser-output"&gt;
 &lt;span typeof="mw:File"&gt;
  &lt;a href="/wiki/Wikipedia:Good_articles*" title="This is a good article. Click here for more information."&gt;
   &lt;img alt="This is a good article. Click here for more information." class="mw-file-element" data-file-height="185" data-file-width="180" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/9/94/Symbol_support_vote.svg/20px-Symbol_support_vote.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/9/94/Symbol_support_vote.svg/40px-Symbol_support_vote.svg.png 1.5x" width="19"/&gt;
  &lt;/a&gt;
 &lt;/span&gt;
&lt;/div&gt;

- It's supposed to be scrape_webpage(url)

- ⚠️ No &lt;p&gt; tags found inside the content container.

- ⚠️ Content div does not contain expected article content.
💾 Saved full HTML to debug_page.html

- The debug_page.html is too long to paste here

- I used this version:

    ```
    import requests
    from bs4 import BeautifulSoup

    def scrape_webpage(url):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }

        try:
            session = requests.Session()
            response = session.get(url, headers=headers)
            if response.status_code != 200:
                print(f"❌ Failed to fetch the webpage. Status code: {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            content_div = soup.find("div", class_="mw-parser-output")

            if not content_div:
                print("❌ Could not find the main content container.")
                return None

            # Look for keyword before continuing
            if "Rage Against the Machine" not in content_div.text:
                print("⚠️ Content div does not contain expected article content.")
                with open("debug_page.html", "w", encoding="utf-8") as debug_file:
                    debug_file.write(soup.prettify())
                print("💾 Saved full HTML to debug_page.html")
                return None

            paragraphs = content_div.find_all("p")
            if not paragraphs:
                print("⚠️ No &lt;p&gt; tags found inside the content container.")
                with open("debug_page.html", "w", encoding="utf-8") as debug_file:
                    debug_file.write(content_div.prettify())
                print("💾 Saved partial HTML to debug_page.html")
                return None

            article_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            with open("Selected_Document.txt", "w", encoding="utf-8") as file:
                file.write(article_text)

            print("✅ Successfully fetched and saved article.")
            return article_text

        except Exception as e:
            print(f"⚠️ An error occurred: {e}")
            return None

    def main():
        url = "https://en.wikipedia.org/wiki/Punk_rock"
        scrape_webpage(url)

    if __name__ == '__main__':
        main()
    ```

- And used a different wiki article and it worked


- The text has some words smushed together with no spaces


- Can we get it to remove all the [ 22 ] and whatnot? I'm guessing they're annotations to citations


- Sometimes it has an extra space where the [number] used to be


- Now it's all on one line, which I don't want


- Sometimes it has something like [ nb 6 ] or [ 6, 2, 3 ]


- There's still stuff like this: [ nb 1 ] 
And I like the previous version more that split it up into paragraphs.

## Step 3 Prompts
- Write code to import logging, transformers.logging (as hf_logging), and warnings; then set the log level of langchain.text_splitter and transformers to ERROR, and filter Python warnings. Add this to the bottom of the existing program.


- Write code to define the variables
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5
and add it to the bottom of the existing program.


- Write code to open Selected_Document.txt in UTF‑8 mode, read its contents into a variable text, and add it to the bottom of the existing program.


- Write code to import and use RecursiveCharacterTextSplitter (with separators ['\n\n', '\n', ' ', ''] and the above chunk_size and chunk_overlap) to split text into a list chunks, and add it to the bottom of the existing program.


- Write code to load SentenceTransformer(model_name), encode chunks (showing a hidden progress bar), convert the result to a NumPy float32 array, initialize a FAISS IndexFlatL2 with the correct dimension, add the array to it, and add this snippet to the bottom of the existing program.


- Write code to import and set up a HuggingFace pipeline('text2text-generation', model='google/flan-t5-small', device=-1), assign it to generator, and add it to the bottom of the existing program.


- Write code to define:

def retrieve_chunks(question, k=top_k):
    # encode the question, search the FAISS index, return top k chunks

def answer_question(question):
    # call retrieve_chunks, build a prompt with context, call generator, and return generated_text
and add these two functions to the bottom of the existing program.



- Write code to wrap an input loop under:

if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit","quit"):
            break
        print("Answer:", answer_question(question))
so that the user can keep asking until they type ‘exit’ or ‘quit’, and add it to the bottom of the existing program.

### Debugging prompts
- Import "transformers.logging" could not be resolvedPylancereportMissingImports
(module) transformers


- How come 'distances' here is never accessed?

```
def retrieve_chunks(question, k=top_k):
    """
    Encode the question, search the FAISS index, and return the top k relevant chunks.
    """
    query_vector = embedding_model.encode([question], show_progress_bar=False)
    query_vector = np.array(query_vector, dtype=np.float32)
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0] if i &lt; len(chunks)]
```



- Your last change to retrieve_chunks did this:

🧠 Retrieval-Augmented Generation App Ready!
Type your questions below. Type 'exit' or 'quit' to stop.

Your question: What is punk rock?
Traceback (most recent call last):
  File "/mnt/c/Users/Morgon/Documents/Full Sail/AI Ecosystem/Retrieval-Augmented_Generation/RAG_app.py", line 106, in &lt;module&gt;
    answer = answer_question(question)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/Morgon/Documents/Full Sail/AI Ecosystem/Retrieval-Augmented_Generation/RAG_app.py", line 93, in answer_question
    context = "\n\n".join(context_chunks)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Step 4 Prompt
Use your AI tool to generate five important questions to understand this RAG system (e.g., about embedding dimensionality, FAISS search behavior, chunk overlap, prompt design) and print both the questions and AI‑generated answers. Include these in your reflection.