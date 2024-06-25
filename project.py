import PyPDF2
from transformers import BertTokenizer, BertForSequenceClassification, pipeline


def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        text = ""
        for page in range(num_pages):
            page_text = reader.pages[page].extract_text()
            if "Risk Consideration" in page_text:
                text += page_text.split("Risk Consideration")[0]
                break
            text += page_text
            
    return text


def summarize_text(text):
    summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum")
    max_input_length = 1024
    chunks = [text[i:i + max_input_length] for i in range(0, len(text), max_input_length)]
    
    summary = ""
    for chunk in chunks:
        summary_text = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += summary_text[0]['summary_text'] + " "
        print(summary)
    
    return summary

# Example usage
file_path = "/Users/Ryan/Desktop/Fixed Income Musings.pdf"  # Replace with your PDF file path
pdf_text = read_pdf(file_path)
summary = summarize_text(pdf_text)

print("Summary:")
print(summary)
