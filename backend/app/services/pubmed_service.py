import requests
from xml.etree import ElementTree as ET
from typing import List, Optional
from decouple import config

class PubMedService:
    """PubMed API wrapper for medical literature search"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config('pubmed_api', default=None)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
    def search(self, query: str, max_results: int = 3) -> List[str]:
        """Search PubMed and return article abstracts"""
        # Search for PMIDs
        search_url = f"{self.base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": max_results
        }
        if self.api_key:
            search_params["api_key"] = self.api_key
            
        try:
            response = requests.get(search_url, params=search_params, timeout=10)
            root = ET.fromstring(response.text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            
            if not pmids:
                return []
            
            # Fetch abstracts
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "text",
                "rettype": "abstract"
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key
                
            response = requests.get(fetch_url, params=fetch_params, timeout=10)
            articles = response.text.split("\n\n")
            
            # Clean and return abstracts
            abstracts = []
            for article in articles:
                lines = article.split("\n")
                abstract_lines = [line for line in lines if line.strip() 
                                and not any(skip in line.lower() for skip in 
                                          ["author", "doi", "pmid", "copyright"])]
                if abstract_lines:
                    abstracts.append(" ".join(abstract_lines))
                    
            return abstracts
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []

pubmed_service = PubMedService()
