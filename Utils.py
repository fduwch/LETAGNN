import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import random
import json
import pandas as pd

class DataSource:
    """
    Ethereum blockchain data retrieval utilities.
    - Wraps Etherscan API for normal, internal, and ERC20 token transfers.
    - Rotates multiple API keys to avoid rate limits.
    """

    def __init__(self):
        self.apikeys = [
            ""
        ]
        self.headers = {
            "content-type": "application/json",
            "user-agent": "",
        }
        self.url_rpc = ""

    def _get_etherscan_data(self, module, action, address, startblock, endblock, page=1, offset=10000, sort="asc", **kwargs):
        """Generic method to fetch data from Etherscan API."""
        params = {
            "chainid": 1,
            "module": module,
            "action": action,
            "address": address,
            "startblock": startblock,
            "endblock": endblock,
            "page": page,
            "offset": offset,
            "sort": sort,
            "apikey": random.choice(self.apikeys)
        }
        params.update(kwargs)
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.etherscan.io/v2/api?{url_params}"
        return getDataFromUrl(url, self.headers).json()["result"]

    def getNormalTransactionsbyAddress(self, address, startblock, endblock, page, offset=10000, sort="asc"):
        """Get a list of 'Normal' transactions by address."""
        return self._get_etherscan_data("account", "txlist", address, startblock, endblock, page, offset, sort)

    def getInternalTransactionsbyAddress(self, address, startblock, endblock, page=1, offset=10000, sort="asc"):
        """Get a list of 'Internal' transactions by address (ETH)."""
        return self._get_etherscan_data("account", "txlistinternal", address, startblock, endblock, page, offset, sort)
    
    def getInternalTransactionsbyTransactionHash(self, txhash):
        """Get a list of 'Internal' transactions by transaction hash."""
        params = {
            "chainid": 1,
            "module": "account",
            "action": "txlistinternal",
            "txhash": txhash,
            "apikey": random.choice(self.apikeys)
        }
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.etherscan.io/v2/api?{url_params}"
        return getDataFromUrl(url, self.headers).json()["result"]

    def getERCTokenTransferbyAddress(self, action, address, startblock, endblock, page, offset=10000, contractaddress="", sort="asc"):
        """
        Get token transfer events by address.
        action ∈ {tokentx, tokennfttx, token1155tx}
        """
        kwargs = {}
        if contractaddress:
            kwargs["contractaddress"] = contractaddress
        return self._get_etherscan_data("account", action, address, startblock, endblock, page, offset, sort, **kwargs)

    def getTotalDatafromScan(self, address, ttype, saved_path, start_number=0, end_number=99999999):
        """Fetch all transactions of a specific type and save to CSV."""
        saved_path_address = f"{saved_path}{address}.csv"
        response_list = []
        
        type_method_map = {
            'Normal/': lambda: self.getNormalTransactionsbyAddress(address, start_number, end_number, 1),
            'Internal/': lambda: self.getInternalTransactionsbyAddress(address, start_number, end_number, 1),
            'ERC20/': lambda: self.getERCTokenTransferbyAddress('tokentx', address, start_number, end_number, 1)
        }
        
        max_attempts = 2
        attempt_count = 0
        
        while attempt_count < max_attempts:
            if ttype not in type_method_map:
                return False, 0
            
            response = type_method_map[ttype]()
            if not response:
                return False, 0
            
            response_list.extend(response)
            
            # If fewer than 10,000 returned, finished.
            if len(response) < 10000:
                if response_list:
                    pd.DataFrame(response_list).to_csv(saved_path_address, index=None)
                return True, len(response_list)
            
            # Advance start block for next page
            start_number = int(response[-1]["blockNumber"])
            attempt_count += 1
        
        # Reached max attempts; save what we have
        if response_list:
            pd.DataFrame(response_list).to_csv(saved_path_address, index=None)
        return True, len(response_list)

    def getTransactionCountfromRPC(self, address):
        """Get transaction count for an address via RPC."""
        payload = {
            "method": "eth_getTransactionCount",
            "params": [address, "latest"],
            "id": 1,
            "jsonrpc": "2.0",
        }
        return int(getDatafromRPC(payload)["result"], 16)
        
    def getBalancefromRPC(self, address):
        """Get ETH balance for an address via RPC."""
        payload = {
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1,
            "jsonrpc": "2.0",
        }
        return int(getDatafromRPC(payload)["result"], 16) / 10**18  # wei -> ETH


def getDataFromUrl(url, headers, data=None, sstype='get', timeout=20):
    """HTTP request with retries and backoff; returns Response or None on failure."""
    retries = Retry(total=10, backoff_factor=0.9)
    
    with requests.Session() as session:
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        try:
            if sstype == 'get':
                response = session.get(url, headers=headers, data=data, timeout=timeout)
            else:
                response = session.post(url, headers=headers, data=data, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            # Fail silently; caller handles None
            return None
            

def getAddressLabelFromEthereumPage(address):
    """
    Scrape Etherscan address page to extract label information.
    Returns (span_text, title_text, label_text).
    """
    headers = {
        'cookie': '__stripe_mid=901553e0-d3df-4424-8b9e-fcffccc0d698055c37; etherscan_offset_datetime=+8; etherscan_cookieconsent=True; etherscan_switch_token_amount_value=value; etherscan_switch_age_datetime=Age; _ga=GA1.1.882709905.1679023127; _ga_T1JC9RNQXV=GS1.1.1739777391.69.1.1739778303.60.0.0; __cflb=02DiuFnsSsHWYH8WqVXaqSXf986r8yFDsA1zoNWuQ6RPr; ASP.NET_SessionId=r5np40os0i1h3hdlekver0ce; cf_clearance=fukKVqTem3TBD..0bzxJsuuNQ_XmGLFkdQHmGi8e3zk-1744614664-1.2.1.1-IjP9nOVD0EyQ_SWh5Zq8nUHBJDPWTo2puZ_z3_iFGwCwEeNkxOz4wLrpPtLZ4hrGdRbUNgJnhl2lBWtevi4PszUA3veNvmy7k9VtarQYwTzqkYUd5DaxWmOAWKVsmyuyTBTXRbOQ.pG.dUx8AYDlclF0xOPeHmobO1DkVtikFNFTAavZ6tC29qFKgTYd5ZTi_M1L1SvBg687ZVf6bLYRzUwnMBmNC53laUGOZhhLxVz72LFKZ4VFFTBDkMP4ot6JE3qMfs_peH0rWb3iQAxi_AXXuDyRROaaHZgf.j97KJ0vyFXyLn6zGvWX3RGIzdgX4G0gbmX79eYqsNpaEhdf59ao_vfrTensmMltqiGi6N48iWKg.7A8d39cdj2zkdNr',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'origin': 'https://etherscan.io'
    }
    url_address = f'https://etherscan.io/address/{address}'
    
    response = getDataFromUrl(url=url_address, headers=headers)
    if not response:
        return "", "", ""
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Target section that contains label spans
    target_section = soup.select_one('body main section:nth-of-type(3) div:nth-of-type(1) div:nth-of-type(1)')
    
    label_text = ""
    span_text = ""
    
    if target_section:
        all_spans = target_section.find_all('span')
        all_span_contents = [span.get_text().strip() for span in all_spans if span.get_text().strip()]
        
        # Prefer Fake_Phishing span if present
        for content in all_span_contents:
            if "Fake_Phishing" in content:
                span_text = content
                break
        
        # If no specific span, join all spans
        if not span_text and all_span_contents:
            unique_contents = list(dict.fromkeys(all_span_contents))
            span_text = ";".join(unique_contents)
        
        # Prefer a span containing 'Phish'
        for content in all_span_contents:
            if "Phish" in content:
                label_text = content
                break
        
        # Fallback to first non-empty span
        if not label_text and all_span_contents:
            label_text = all_span_contents[0]
    
    # Extract title text
    title_text = ""
    if soup.title:
        title_parts = soup.title.string.split("|")
        if len(title_parts) >= 3:
            title_text = title_parts[0].strip().split("\n")[0]
    
    return span_text, title_text, label_text
            

def getDatafromRPC(payload):
    """Send a JSON-RPC request and return parsed JSON or None."""
    url_rpc = ""
    headers = {
        "content-type": "application/json",
        "user-agent": "",
    }
    response = getDataFromUrl(url_rpc, headers=headers, data=json.dumps(payload), sstype='post')
    return response.json() if response else None

if __name__ == "__main__":
    data_source = DataSource()
    address = '0xdAC17F958D2ee523a2206206994597C13D831ec7'
    print(getAddressLabelFromEthereumPage(address))