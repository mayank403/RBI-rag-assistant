"""
RBI Data Scraper - Extracts circulars, notifications, and press releases from rbi.org.in
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL = "https://www.rbi.org.in"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Rate limiting - be respectful to RBI servers
REQUEST_DELAY = 2  # seconds between requests


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_page(url, retries=3):
    """Fetch a page with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"  [WARN] Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(REQUEST_DELAY * 2)
    return None


def scrape_press_releases(max_pages=3):
    """Scrape press releases from RBI website."""
    print("\n[>] Scraping Press Releases...")
    documents = []

    url = f"{BASE_URL}/Scripts/BS_PressReleaseDisplay.aspx"
    html = fetch_page(url)
    if not html:
        print("  [ERR] Failed to fetch press releases page")
        return documents

    soup = BeautifulSoup(html, "lxml")

    # Find all press release links
    links = soup.find_all("a", class_="link2")
    print(f"  [>] Found {len(links)} press release links")

    for i, link in enumerate(links[:50]):  # Limit to 50 for prototype
        title = link.get_text(strip=True)
        href = link.get("href", "")

        if not href or "prid=" not in href:
            continue

        prid = re.search(r"prid=(\d+)", href)
        if not prid:
            continue

        detail_url = f"{BASE_URL}/Scripts/{href}"
        print(f"  [>] [{i+1}] Fetching: {title[:60]}...")

        detail_html = fetch_page(detail_url)
        if not detail_html:
            continue

        detail_soup = BeautifulSoup(detail_html, "lxml")

        # Extract content from the press release detail page
        content_div = detail_soup.find("div", class_="text1")
        if content_div:
            # Remove scripts and styles
            for tag in content_div.find_all(["script", "style"]):
                tag.decompose()
            content = content_div.get_text(separator="\n", strip=True)
        else:
            content = title

        # Try to extract date
        date_text = ""
        date_elem = detail_soup.find("td", class_="tableheader")
        if date_elem:
            date_text = date_elem.get_text(strip=True)

        doc = {
            "id": f"pr_{prid.group(1)}",
            "title": title,
            "content": content,
            "date": date_text,
            "url": detail_url,
            "type": "press_release",
            "source": "RBI",
        }
        documents.append(doc)
        time.sleep(REQUEST_DELAY)

    return documents


def scrape_notifications(max_items=30):
    """Scrape notifications from RBI website."""
    print("\n[>] Scraping Notifications...")
    documents = []

    url = f"{BASE_URL}/Scripts/NotificationUser.aspx"
    html = fetch_page(url)
    if not html:
        print("  [ERR] Failed to fetch notifications page")
        return documents

    soup = BeautifulSoup(html, "lxml")

    # Find notification links
    links = soup.find_all("a", class_="link2")
    print(f"  [>] Found {len(links)} notification links")

    for i, link in enumerate(links[:max_items]):
        title = link.get_text(strip=True)
        href = link.get("href", "")

        if not href:
            continue

        if href.startswith("http"):
            detail_url = href
        elif href.startswith("/"):
            detail_url = f"{BASE_URL}{href}"
        else:
            detail_url = f"{BASE_URL}/Scripts/{href}"

        print(f"  [>] [{i+1}] Fetching: {title[:60]}...")

        detail_html = fetch_page(detail_url)
        if not detail_html:
            continue

        detail_soup = BeautifulSoup(detail_html, "lxml")

        # Extract content
        content_div = detail_soup.find("div", class_="text1")
        if content_div:
            for tag in content_div.find_all(["script", "style"]):
                tag.decompose()
            content = content_div.get_text(separator="\n", strip=True)
        else:
            # Try other content containers
            main_content = detail_soup.find("div", id="mainsection")
            if main_content:
                content = main_content.get_text(separator="\n", strip=True)
            else:
                content = title

        doc = {
            "id": f"notif_{i}",
            "title": title,
            "content": content,
            "date": "",
            "url": detail_url,
            "type": "notification",
            "source": "RBI",
        }
        documents.append(doc)
        time.sleep(REQUEST_DELAY)

    return documents


def scrape_master_directions(max_items=20):
    """Scrape master directions from RBI website."""
    print("\n[>] Scraping Master Directions...")
    documents = []

    url = f"{BASE_URL}/Scripts/BS_ViewMasterDirections.aspx"
    html = fetch_page(url)
    if not html:
        print("  [ERR] Failed to fetch master directions page")
        return documents

    soup = BeautifulSoup(html, "lxml")

    links = soup.find_all("a", class_="link2")
    print(f"  [>] Found {len(links)} master direction links")

    for i, link in enumerate(links[:max_items]):
        title = link.get_text(strip=True)
        href = link.get("href", "")

        if not href:
            continue

        if href.startswith("http"):
            detail_url = href
        elif href.startswith("/"):
            detail_url = f"{BASE_URL}{href}"
        else:
            detail_url = f"{BASE_URL}/Scripts/{href}"

        print(f"  [>] [{i+1}] Fetching: {title[:60]}...")

        detail_html = fetch_page(detail_url)
        if not detail_html:
            continue

        detail_soup = BeautifulSoup(detail_html, "lxml")

        content_div = detail_soup.find("div", class_="text1")
        if content_div:
            for tag in content_div.find_all(["script", "style"]):
                tag.decompose()
            content = content_div.get_text(separator="\n", strip=True)
        else:
            content = title

        doc = {
            "id": f"md_{i}",
            "title": title,
            "content": content,
            "date": "",
            "url": detail_url,
            "type": "master_direction",
            "source": "RBI",
        }
        documents.append(doc)
        time.sleep(REQUEST_DELAY)

    return documents


def save_documents(documents, filename="rbi_documents.json"):
    """Save scraped documents to JSON file."""
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved {len(documents)} documents to {filepath}")
    return filepath


def load_documents(filename="rbi_documents.json"):
    """Load documents from JSON file."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def create_sample_data():
    """Create sample RBI data for immediate testing (no scraping needed)."""
    print("\n[>] Creating sample RBI data for testing...")

    sample_docs = [
        {
            "id": "pr_sample_1",
            "title": "Monetary Policy Statement, 2024-25 Resolution of the Monetary Policy Committee (MPC)",
            "content": """Monetary Policy Statement, 2024-25 Resolution of the Monetary Policy Committee (MPC)

On the basis of an assessment of the current and evolving macroeconomic situation, the Monetary Policy Committee (MPC) at its meeting today (October 9, 2024) decided to:

Keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent.

Consequently, the standing deposit facility (SDF) rate remains unchanged at 6.25 per cent and the marginal standing facility (MSF) rate and the Bank Rate at 6.75 per cent.

The MPC also decided to change the stance from withdrawal of accommodation to neutral and to remain unambiguously focused on a durable alignment of inflation with the target, while supporting growth.

These decisions are in consonance with the objective of achieving the medium-term target for consumer price index (CPI) inflation of 4 per cent within a band of +/- 2 per cent, while supporting growth.

The MPC noted that the Indian economy remains resilient. Real GDP growth for 2024-25 is projected at 7.2 per cent. CPI inflation for 2024-25 is projected at 4.5 per cent.

Key decisions:
1. Policy repo rate unchanged at 6.50%
2. Standing deposit facility rate at 6.25%
3. Marginal standing facility rate at 6.75%
4. Bank Rate at 6.75%
5. Stance changed to neutral from withdrawal of accommodation""",
            "date": "Oct 09, 2024",
            "url": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=58800",
            "type": "press_release",
            "source": "RBI",
        },
        {
            "id": "pr_sample_2",
            "title": "RBI Regulatory Sandbox - Enabling Framework",
            "content": """Enabling Framework for Regulatory Sandbox

The Reserve Bank has set up a Regulatory Sandbox (RS) to enable live testing of new products, services, and business models in a controlled regulatory environment. The framework aims to foster responsible innovation in financial services, promote efficiency and bring benefit to consumers.

Key Features of the Regulatory Sandbox:
1. The RS shall operate within a well-defined space and duration.
2. The RS shall be applicable for innovative products/services/technology in areas such as retail payments, money transfer services, marketplace lending, digital KYC, financial advisory services, wealth management services, digital banks, cybersecurity products, and blockchain/DLT applications.
3. Entities applying for RS include fintech companies registered and operating in India, banks, NBFCs, and other financial entities.
4. The sandbox process involves: Application → Evaluation → Testing → Exit.

Eligibility Criteria:
- The entity must be incorporated and registered in India
- Minimum net worth of ₹25 lakh
- Must have a satisfactory compliance record
- The product/service should be genuinely innovative
- Must demonstrate clear consumer benefit

Regulatory Sandbox Cohorts:
- Cohort 1: Retail Payments
- Cohort 2: Cross Border Payments
- Cohort 3: MSME Lending
- Cohort 4: Prevention and Mitigation of Financial Frauds
- On-tap application for all themes

Duration: Testing phase limited to 6 months, extendable by another 6 months.

Exit: Entities may exit the sandbox by obtaining necessary regulatory approvals or by discontinuing the product/service.""",
            "date": "Aug 13, 2019",
            "url": "https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?UrlPage=&ID=938",
            "type": "notification",
            "source": "RBI",
        },
        {
            "id": "pr_sample_3",
            "title": "Digital Lending Guidelines - RBI Circular",
            "content": """Guidelines on Digital Lending

The Reserve Bank of India issued comprehensive guidelines on digital lending to address concerns related to unbridled engagement of third parties, mis-selling, breach of data privacy, unfair business conduct, charging of exorbitant interest rates, and unethical recovery practices.

Key Provisions:
1. All loan disbursals and repayments must be executed only between the bank accounts of the borrower and the Regulated Entity (RE) without any pass-through/pool account of the Lending Service Provider (LSP) or any third party.

2. Fees/charges payable to LSPs shall be paid directly by the RE and not by the borrower.

3. A standardized Key Fact Statement (KFS) must be provided to the borrower before executing the loan agreement. The KFS must contain:
   - All-inclusive cost of digital loans in the form of Annual Percentage Rate (APR)
   - Recovery mechanism
   - Details of grievance redressal officer

4. Data Collection: Digital lending apps can only collect data with prior and explicit consent of the borrower. Access to mobile phone resources like file and media, contact list, call logs, telephony functions is prohibited.

5. Cooling-off Period: A cooling-off/look-up period must be provided during which the borrower can exit the loan without penalty.

6. Grievance Redressal: Each RE must ensure that they and their LSPs/DLAs have a suitable nodal grievance redressal officer to deal with fintech/digital lending related complaints.

These guidelines apply to all commercial banks, primary (urban) co-operative banks, state co-operative banks, district central co-operative banks, and NBFCs (including housing finance companies).""",
            "date": "Sep 02, 2022",
            "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12382",
            "type": "notification",
            "source": "RBI",
        },
        {
            "id": "pr_sample_4",
            "title": "Master Direction on KYC - Know Your Customer Direction",
            "content": """Master Direction - Know Your Customer (KYC) Direction, 2016

The Reserve Bank of India has issued comprehensive directions on Know Your Customer (KYC) norms/Anti-Money Laundering (AML) Standards/Combating of Financing of Terrorism (CFT)/Obligation of Regulated Entities under the Prevention of Money Laundering Act (PMLA), 2002.

Key Provisions:

1. Customer Due Diligence (CDD):
   - Regulated entities (REs) must carry out CDD while establishing an account-based relationship
   - Verify identity of the customer using OVDs (Officially Valid Documents)
   - PAN or Form 60 is mandatory for specified transactions

2. Simplified KYC:
   - For low-risk customers with balances not exceeding ₹50,000
   - Small accounts can be opened with self-declaration and one recent photograph
   - Aadhaar-based e-KYC is acceptable for simplified due diligence

3. Video-based Customer Identification Process (V-CIP):
   - REs may undertake Video-based Customer Identification Process
   - Must be a seamless, real-time, secured, end-to-end encrypted audio-video interaction
   - Customer's live photograph and identification details must be captured

4. Central KYC Records Registry (CKYCR):
   - All REs must upload KYC data to CKYCR
   - 14-digit KYC Identification Number assigned to each customer

5. Periodic Updation:
   - High risk customers: every 2 years
   - Medium risk customers: every 8 years
   - Low risk customers: every 10 years

6. Digital KYC:
   - Aadhaar OTP-based e-KYC permitted for accounts up to ₹10,000 balance
   - DigiLocker documents accepted as OVDs
   - CKYC can be used for simplified on-boarding""",
            "date": "Feb 25, 2016",
            "url": "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx?id=11566",
            "type": "master_direction",
            "source": "RBI",
        },
        {
            "id": "pr_sample_5",
            "title": "UPI - Unified Payments Interface Guidelines",
            "content": """Unified Payments Interface (UPI) - Guidelines and Framework

The Reserve Bank of India and NPCI have established the Unified Payments Interface (UPI) as a real-time payment system facilitating inter-bank peer-to-peer (P2P) and person-to-merchant (P2M) transactions.

Key Features and Guidelines:

1. Transaction Limits:
   - UPI transaction limit: ₹1,00,000 per transaction (general)
   - Enhanced limit of ₹2,00,000 for specific categories
   - ₹5,00,000 for capital markets, insurance, and collections
   - UPI Lite: up to ₹500 per transaction, wallet limit ₹2,000

2. UPI Lite:
   - Near-offline small value payments
   - No UPI PIN required for transactions up to ₹500
   - Auto-top up facility available
   - Real-time notification for each debit

3. UPI 123PAY (Feature Phone):
   - IVR (Interactive Voice Response) based
   - App-based using Java
   - Missed call based
   - Sound-based proximity payments

4. Third Party Application Providers (TPAPs):
   - Must partner with banks as Payment System Providers (PSPs)
   - Must comply with data localization requirements
   - Customer data must be stored in India

5. Merchant Payments:
   - QR code-based payments (static and dynamic)
   - Interoperable QR codes across payment systems
   - No merchant discount rate (MDR) on UPI transactions

6. Security Features:
   - Device binding and UPI PIN for authentication
   - Two-factor authentication for transactions above ₹200
   - Fraud monitoring and reporting mechanisms

7. Dispute Resolution:
   - Technical decline: auto-reversal within T+5 business days
   - Online Dispute Resolution (ODR) system for failed transactions
   - RBI Ombudsman scheme covers UPI complaints""",
            "date": "Mar 15, 2023",
            "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12500",
            "type": "notification",
            "source": "RBI",
        },
        {
            "id": "pr_sample_6",
            "title": "RBI Guidelines on Outsourcing of Financial Services",
            "content": """Guidelines on Managing Risks and Code of Conduct in Outsourcing of Financial Services by Banks

The Reserve Bank of India has issued guidelines governing outsourcing of financial services by banks. These guidelines aim to ensure that outsourcing arrangements neither diminish the ability of banks to fulfil their obligations to customers nor impede effective supervision by the RBI.

Key Provisions:

1. Activities Not Permitted to be Outsourced:
   - Core management functions including internal audit, compliance, KYC
   - Strategic and compliance functions
   - Decision-making functions relating to credit management

2. Risk Management Framework:
   - Banks must have a comprehensive outsourcing policy approved by the Board
   - Due diligence of service providers before engagement
   - Regular monitoring and review of outsourced activities

3. Confidentiality and Security:
   - Service providers must maintain confidentiality of customer data
   - Banks remain responsible for security of customer information
   - Data must be stored in India for critical operations

4. Business Continuity:
   - Outsourcing arrangements should not affect continuity of services
   - Banks must have contingency plans for failure of service providers
   - Exit strategy must be defined in agreements

5. Cross-Border Outsourcing:
   - Prior approval from RBI for outsourcing critical functions overseas
   - Regulatory access to books and records must be maintained
   - Compliance with Indian laws and regulations""",
            "date": "Nov 03, 2006",
            "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=3148",
            "type": "notification",
            "source": "RBI",
        },
        {
            "id": "pr_sample_7",
            "title": "NBFC Registration and Regulation Guidelines",
            "content": """Non-Banking Financial Company (NBFC) - Registration and Regulatory Framework

The Reserve Bank of India regulates NBFCs under the provisions of the Reserve Bank of India Act, 1934, Chapter III-B, III-C, and V.

Key Regulatory Provisions:

1. Registration Requirements:
   - Minimum Net Owned Fund (NOF): ₹10 crore (enhanced from ₹2 crore)
   - Company incorporated under Companies Act, 2013
   - Principal business of NBFC: financial activity constituting >50% of total assets and >50% of gross income
   - Must obtain Certificate of Registration (CoR) from RBI

2. Scale-Based Regulation (SBR) Framework:
   - Base Layer (NBFC-BL): NBFCs with asset size below ₹1,000 crore
   - Middle Layer (NBFC-ML): All deposit-taking NBFCs, NBFCs with asset size ≥ ₹1,000 crore
   - Upper Layer (NBFC-UL): Top 10 NBFCs by asset size and systemically important entities
   - Top Layer (NBFC-TL): Only if substantial increase in systemic risk

3. Capital Requirements:
   - CRAR: Minimum 15% (including CCB of 2.5%)
   - Tier 1 capital: Minimum 10%
   - Leverage ratio: Maximum 7x for NBFC-UL

4. Prudential Norms:
   - NPA classification: 90 days overdue
   - Provisioning norms aligned with banks
   - Fair Practices Code mandatory
   - Interest rate transparency

5. Digital NBFCs:
   - Must comply with digital lending guidelines
   - First Loss Default Guarantee (FLDG) limited to 5%
   - All disbursals through bank accounts
   - KFS mandatory for all loans""",
            "date": "Oct 22, 2021",
            "url": "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx?id=12100",
            "type": "master_direction",
            "source": "RBI",
        },
        {
            "id": "pr_sample_8",
            "title": "Foreign Exchange Management - Current Account Transactions",
            "content": """Foreign Exchange Management (Current Account Transactions) Rules

The Reserve Bank of India, under the Foreign Exchange Management Act (FEMA), 1999, has issued rules governing current account transactions in foreign exchange.

Key Provisions:

1. Liberalized Remittance Scheme (LRS):
   - Resident individuals can remit up to USD 250,000 per financial year
   - Permitted for education, medical treatment, travel, gifts, donations
   - Investment in shares, debt instruments, and property abroad
   - Opening of foreign currency accounts abroad

2. Prohibited Transactions:
   - Remittance for purchase of lottery tickets
   - Remittance for income from racing/riding
   - Remittance for purchase of FCCBs issued by Indian companies in overseas secondary market
   - Remittance for trading in foreign exchange abroad
   - Capital account remittances directly or indirectly to countries identified as non-cooperative by FATF

3. Export of Goods and Services:
   - Declaration on appropriate forms (SB for customs, SOFTEX for software)
   - Realization of export proceeds within 9 months
   - Advance remittance for export of goods permitted
   - Export factoring permitted through authorized dealers

4. Import Payments:
   - Payment through normal banking channels
   - Advance payment permitted with bank guarantees above USD 200,000
   - Import of gold restricted to authorized entities
   - Trade credits governed by separate guidelines

5. Business Travel:
   - Basic Travel Quota: USD 250,000 under LRS
   - Forex for business travel through authorized dealers
   - International credit/debit cards for current account transactions""",
            "date": "May 03, 2000",
            "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=100",
            "type": "notification",
            "source": "RBI",
        },
        {
            "id": "pr_sample_9",
            "title": "RBI Framework for Acceptance of Deposits by NBFCs",
            "content": """Framework for Acceptance of Public Deposits by NBFCs

The Reserve Bank of India has laid down a comprehensive framework governing the acceptance of public deposits by Non-Banking Financial Companies.

Key Provisions:

1. Eligibility:
   - Only NBFCs with specific authorization from RBI can accept public deposits
   - Minimum investment grade credit rating required
   - Net Owned Fund of at least ₹25 lakh (now ₹10 crore)
   - Compliance with all prudential norms

2. Deposit Limits:
   - Maximum 1.5 times of NOF for NBFCs with investment grade rating
   - Maximum 4 times of NOF for AFCs (Asset Finance Companies) with certain ratings
   - Maximum tenure: 60 months; Minimum tenure: 12 months

3. Interest Rate:
   - Maximum interest rate on public deposits: 12.5% per annum (ceiling)
   - Brokerage: Maximum 2% of deposits mobilized
   - RBI reviews ceiling rates periodically

4. Deposit Insurance:
   - Unlike bank deposits, NBFC deposits are NOT covered under DICGC
   - NBFCs must clearly disclose this to depositors

5. Asset-Liability Management:
   - NBFCs accepting deposits must have robust ALM framework
   - Structural liquidity and interest rate sensitivity statements
   - Regular stress testing and monitoring
   - Board-approved ALM policy mandatory

6. Returns and Reporting:
   - Quarterly NBS-1 return on deposits
   - Annual NBS-3 return
   - Certificate from statutory auditors regarding deposit compliance""",
            "date": "Jul 01, 2020",
            "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11900",
            "type": "notification",
            "source": "RBI",
        },
        {
            "id": "pr_sample_10",
            "title": "Central Bank Digital Currency (CBDC) - Digital Rupee Pilot",
            "content": """Central Bank Digital Currency (CBDC) - e₹ Digital Rupee

The Reserve Bank of India launched pilot projects for both wholesale (e₹-W) and retail (e₹-R) Central Bank Digital Currency.

Key Features:

1. Digital Rupee - Retail (e₹-R):
   - Legal tender issued by RBI in digital form
   - Denominated in the same way as sovereign currency
   - Exchangeable at par with existing currency
   - Token-based for anonymity in small transactions

2. Architecture:
   - Two-tier model: RBI issues CBDC to intermediaries who distribute to users
   - Intermediaries include banks and other authorized entities
   - Based on Distributed Ledger Technology (DLT)

3. Pilot Details:
   - Wholesale pilot (e₹-W): Launched November 1, 2022 for settlement of secondary market G-Sec transactions
   - Retail pilot (e₹-R): Launched December 1, 2022 in select cities
   - Participating banks: SBI, ICICI Bank, Yes Bank, IDFC First Bank, Bank of Baroda, Union Bank of India, HDFC Bank, Kotak Mahindra Bank

4. Features:
   - Person-to-person (P2P) and person-to-merchant (P2M) transactions
   - QR code-based payments
   - Near-offline capability for small transactions
   - Programmability for targeted subsidies and benefits

5. Privacy and Security:
   - Tiered anonymity based on transaction value
   - Small value transactions: cash-like anonymity
   - Large value transactions: identity verification required
   - Encrypted transactions ensuring data security

6. Impact on Monetary Policy:
   - Direct tool for monetary policy transmission
   - Potential for negative interest rates if needed
   - Better tracking of money supply
   - Cross-border payment efficiency""",
            "date": "Oct 07, 2022",
            "url": "https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?UrlPage=&ID=1218",
            "type": "press_release",
            "source": "RBI",
        },
        {
            "id": "pr_sample_11",
            "title": "Priority Sector Lending - Targets and Classification",
            "content": """Priority Sector Lending (PSL) - Targets and Classification

The Reserve Bank of India mandates all scheduled commercial banks and foreign banks to lend a specified portion of their Adjusted Net Bank Credit (ANBC) to priority sectors.

Key Provisions:

1. Targets:
   - Domestic banks and foreign banks (with 20+ branches): 40% of ANBC
   - Foreign banks (with less than 20 branches): 40% of ANBC (to be achieved in phases)
   - Regional Rural Banks and Small Finance Banks: 75% of ANBC

2. Sub-targets:
   - Agriculture: 18% of ANBC (of which 10% to Small and Marginal Farmers)
   - Micro Enterprises: 7.5% of ANBC
   - Advances to Weaker Sections: 12% of ANBC

3. Categories under Priority Sector:
   - Agriculture and Allied Activities
   - Micro, Small and Medium Enterprises (MSMEs)
   - Export Credit
   - Education (up to ₹20 lakh)
   - Housing (up to ₹35 lakh in metropolitan areas, ₹25 lakh in other areas)
   - Social Infrastructure (schools, healthcare, drinking water, sanitation)
   - Renewable Energy (up to ₹30 crore)
   - Others (including Startups up to ₹50 crore)

4. Priority Sector Lending Certificates (PSLCs):
   - Tradable certificates enabling banks to meet PSL shortfalls
   - Four categories: PSLC-Agriculture, PSLC-SF/MF, PSLC-Micro, PSLC-General
   - Trading through e-Kuber platform of RBI

5. Non-Achievement Penalty:
   - Shortfall invested in RIDF with NABARD, SIDBI, NHB, or MUDRA
   - Interest rate on such deposits lower than market rate
   - Mandatory reporting of PSL achievement quarterly""",
            "date": "Sep 04, 2020",
            "url": "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx?id=11959",
            "type": "master_direction",
            "source": "RBI",
        },
        {
            "id": "pr_sample_12",
            "title": "RBI Account Aggregator Framework",
            "content": """Account Aggregator Framework - RBI Guidelines

The Reserve Bank of India introduced the Account Aggregator (AA) framework to enable flow-based lending and financial data sharing with customer consent.

Key Features:

1. Definition:
   - Account Aggregators are RBI-regulated NBFCs
   - They facilitate sharing of financial data between Financial Information Providers (FIPs) and Financial Information Users (FIUs)
   - They act as consent managers - they don't store or process data themselves

2. Participants:
   - Financial Information Providers (FIPs): Banks, NBFCs, Insurance companies, Pension funds, Mutual funds
   - Financial Information Users (FIUs): Lending institutions, Insurance companies, Wealth managers
   - Account Aggregators: RBI-registered NBFC-AAs

3. Data Sharing Process:
   - Customer provides explicit consent through AA
   - Consent artifact specifies: purpose, duration, frequency, data types
   - Data shared is encrypted end-to-end
   - AA cannot view, store, or sell the data

4. Use Cases:
   - Loan underwriting using bank statements and GST data
   - Insurance underwriting using financial history
   - Personal financial management dashboards
   - Tax filing and compliance
   - Investment advisory using complete financial picture

5. Technical Standards:
   - Based on ReBIT (Reserve Bank Information Technology) standards
   - Data shared in machine-readable format
   - RESTful APIs for data exchange
   - FI (Financial Information) types include banking, insurance, securities, pensions, tax

6. Consumer Protection:
   - Consent must be explicit, informed, and revocable
   - Granular control over data sharing
   - Right to withdraw consent at any time
   - Grievance redressal mechanism mandatory""",
            "date": "Sep 02, 2021",
            "url": "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx?id=12032",
            "type": "master_direction",
            "source": "RBI",
        },
    ]

    return sample_docs


def run_scraper(use_sample=True, scrape_live=False):
    """Main function to run the scraper."""
    ensure_data_dir()

    all_documents = []

    if use_sample:
        # Use sample data for immediate testing
        sample_docs = create_sample_data()
        all_documents.extend(sample_docs)
        print(f"\n[OK] Total sample documents: {len(all_documents)}")

    if scrape_live:
        # Scrape live data from RBI
        print("\n[>] Starting live scraping from RBI website...")
        print("[>] This may take several minutes due to rate limiting...\n")

        press_releases = scrape_press_releases()
        all_documents.extend(press_releases)

        notifications = scrape_notifications()
        all_documents.extend(notifications)

        master_directions = scrape_master_directions()
        all_documents.extend(master_directions)

        print(f"\n[OK] Total live documents scraped: {len(all_documents)}")

    # Save all documents
    filepath = save_documents(all_documents)
    return filepath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RBI Data Scraper")
    parser.add_argument(
        "--live", action="store_true", help="Scrape live data from RBI website"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        default=True,
        help="Use only sample data (default)",
    )
    args = parser.parse_args()

    run_scraper(use_sample=True, scrape_live=args.live)
