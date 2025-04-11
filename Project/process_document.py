import fitz  # PyMuPDF
import json
import os

def extract_text_with_metadata(pdf_path, heading_threshold=14):
    """
    Extract full text from a PDF along with page numbers and potential section headings.
    
    Args:
        pdf_path (str): Path to the PDF file.
        heading_threshold (float): Font size threshold to consider text as a heading.

    Returns:
        list of dict: Each dict contains page_number, headings (if detected), and full page text.
    """
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page in doc:
        page_number = page.number + 1  # converting 0-indexed to human-friendly 1-indexed
        page_dict = page.get_text("dict")
        page_text_lines = []
        headings = []
        
        # Process each block in the page
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue  # Skip non-text blocks
            block_text = ""
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    # Append text and space
                    line_text += text + " "
                    # Heuristic: treat spans with larger font size as headings
                    if span["size"] >= heading_threshold:
                        if text not in headings:
                            headings.append(text)
                block_text += line_text.strip() + "\n"
            page_text_lines.append(block_text.strip())
        
        page_data = {
            "page_number": page_number,
            "headings": headings,
            "text": "\n".join(page_text_lines)
        }
        pages_data.append(page_data)
    
    return pages_data

def segment_page_text(page_data):
    """
    Segment the text of a single page using detected section headers as splitting markers.
    
    Args:
        page_data (dict): Dictionary containing 'page_number', 'headings', and full 'text'
        
    Returns:
        list of dict: Each dictionary represents a text chunk with metadata.
    """
    page_number = page_data["page_number"]
    text = page_data["text"]
    headings = page_data.get("headings", [])
    chunks = []

    # If headings were detected, split text by headings
    if headings:
        lines = text.splitlines()
        current_section = None
        current_chunk_lines = []

        for line in lines:
            # If line is exactly one of the detected headings, consider it as a section marker.
            if line.strip() in headings:
                # If existing section, save it.
                if current_section is not None and current_chunk_lines:
                    chunks.append({
                        "page_number": page_number,
                        "section_title": current_section,
                        "chunk_text": "\n".join(current_chunk_lines).strip()
                    })
                # Update current section and reset accumulator.
                current_section = line.strip()
                current_chunk_lines = []
            else:
                current_chunk_lines.append(line)
                
        # Save the final section if exists.
        if current_section is not None and current_chunk_lines:
            chunks.append({
                "page_number": page_number,
                "section_title": current_section,
                "chunk_text": "\n".join(current_chunk_lines).strip()
            })
        # If no section was created, save entire page as one chunk.
        if not chunks:
            chunks.append({
                "page_number": page_number,
                "section_title": headings[0],
                "chunk_text": text
            })
    else:
        # For pages with no headings, treat the full page as one chunk.
        chunks.append({
            "page_number": page_number,
            "section_title": f"Page {page_number}",
            "chunk_text": text
        })

    return chunks

def segment_all_pages(pages_data):
    """
    Process all pages to segment text into chunks based on section headers.
    
    Args:
        pages_data (list): List of page dictionaries from PDF extraction.
        
    Returns:
        list of dict: A flat list of all text chunks with metadata.
    """
    all_chunks = []
    for page in pages_data:
        chunks = segment_page_text(page)
        all_chunks.extend(chunks)
    return all_chunks

def main(pdf_path, output_json="catalog_chunks.json", heading_threshold=14):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' does not exist.")
        return
    
    # Step 1: PDF Parsing and Text Extraction
    print("Extracting text and metadata from PDF...")
    pages_data = extract_text_with_metadata(pdf_path, heading_threshold)
    print(f"Extracted data from {len(pages_data)} pages.")
    
    # Step 2: Text Segmentation and Metadata Attachment
    print("Segmenting extracted text by section headers...")
    catalog_chunks = segment_all_pages(pages_data)
    print(f"Total chunks created: {len(catalog_chunks)}")
    
    # Save results to a JSON file
    with open(output_json, "w", encoding="utf-8") as outfile:
        json.dump(catalog_chunks, outfile, indent=4, ensure_ascii=False)
    
    print(f"Segmented data saved to {output_json}")

if __name__ == "__main__":
    # Path to your AUI academic catalog PDF.
    pdf_path = "AUI_Catalog_2023_2024_New_Version.pdf"  # Replace with your actual PDF file path.
    main(pdf_path)