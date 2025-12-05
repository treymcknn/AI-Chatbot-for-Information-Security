from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_6e2JzS_PZYtxdavrAwT6JVaBzFVB2gNs4j6DVeu8LQwNf8WfoT3bwmrbQSCWpHHBvWh91P")
index_name = "infosec-policies"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 output dimension
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

index = pc.Index(index_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# cybersecurity policies
policies = [
    {
        "id": "rec49",
        "text": "Strong passwords are long, the more characters a password has the stronger it is. We recommend a minimum of 16 characters in all work related passwords. In addition, we encourage the use of passphrases, passwords made up of multiple words. Examples include 'It's time for vacation' or 'block-curious-sunny-leaves'. Passphrases are both easy to remember and type yet meet the strength requirements.",
        "metadata": {
            "section": "Password Strength Policy",
            "category": "Password Management"
        }
    },
    {
        "id": "rec50",
        "text": "Password cracking or guessing may be performed on a periodic or random basis by the Infosec Team or its delegates. If a password is guessed or cracked during one of these scans, the user will be required to change.",
        "metadata": {
            "section": "Password Testing Policy",
            "category": "Password Management"
        }
    },
    {
        "id": "rec51",
        "text": "Our Network Device Management Policy aims to establish a comprehensive framework for the secure configuration, monitoring, and management of network devices within our organization. ...",
        "metadata": {
            "section": "Network Device Management Overview",
            "category": "Network Security"
        }
    },
    {
        "id": "rec52",
        "text": "NDM-01 Maintain an inventory of each of the organization's approved network devices. NDM-02 Maintain network device cybersecurity configuration benchmarks ...",
        "metadata": {
            "section": "Network Device Management Requirements",
            "category": "Network Security"
        }
    },
    {
        "id": "rec53",
        "text": "Our Data Privacy Policy aims to establish a comprehensive framework for protecting the privacy and confidentiality of personal and sensitive data ...",
        "metadata": {
            "section": "Data Privacy Policy Overview",
            "category": "Data Privacy"
        }
    },
    {
        "id": "rec54",
        "text": "PRV-01 Maintain a transparent, documented privacy program that documents the organization's safeguards to address data privacy. ...",
        "metadata": {
            "section": "Data Privacy Program Requirements",
            "category": "Data Privacy"
        }
    },
    {
        "id": "rec55",
        "text": "SQL Injection is a type of attack where malicious SQL statements are inserted into an entry field for execution.",
        "metadata": {
            "section": "SQL Injection",
            "category": "Application Security"
        }
    },
    {
        "id": "rec56",
        "text": "An Intrusion Detection System (IDS) is a system that detects and alerts on potential security breaches or malicious activities within a network.",
        "metadata": {
            "section": "Intrusion Detection System (IDS)",
            "category": "Network Security"
        }
    },
    {
        "id": "rec57",
        "text": "An Intrusion Prevention System (IPS) is designed to automatically block or prevent detected intrusions from succeeding.",
        "metadata": {
            "section": "Intrusion Prevention System (IPS)",
            "category": "Network Security"
        }
    },
    {
        "id": "rec58",
        "text": "Security Governance is a framework that outlines how an organization manages and governs cybersecurity risk.",
        "metadata": {
            "section": "Security Governance",
            "category": "Governance & Compliance"
        }
    },
    {
        "id": "rec59",
        "text": "Data Exfiltration is the unauthorized copying, transfer, or use of sensitive data outside an organization's boundaries.",
        "metadata": {
            "section": "Data Exfiltration",
            "category": "Data Protection"
        }
    },
    {
        "id": "rec60",
        "text": "Spoofing is a fake website or service designed to trick users into revealing sensitive information.",
        "metadata": {
            "section": "Spoofing",
            "category": "Social Engineering"
        }
    },
    {
        "id": "rec61",
        "text": "A Zero-Day Attack targets software vulnerabilities before developers have released a patch or fix.",
        "metadata": {
            "section": "Zero-Day Attack",
            "category": "Vulnerability Management"
        }
    },
    {
        "id": "rec62",
        "text": "Confidentiality is the process of ensuring that data is accessible only to those authorized to view it.",
        "metadata": {
            "section": "Confidentiality",
            "category": "Information Security Principles"
        }
    },
    {
        "id": "rec63",
        "text": "Cloud Security is the discipline focused on protecting cloud-based data, applications, and infrastructure.",
        "metadata": {
            "section": "Cloud Security",
            "category": "Cloud Computing"
        }
    },
    {
        "id": "rec64",
        "text": "An Information Security Policy is a document that defines an organization's security practices, roles, and responsibilities.",
        "metadata": {
            "section": "Information Security Policy",
            "category": "Governance & Compliance"
        }
    },
    {
        "id": "rec65",
        "text": "A Security Patch is a software update designed to fix vulnerabilities or bugs in applications and systems.",
        "metadata": {
            "section": "Security Patch",
            "category": "Patch Management"
        }
    },
    {
        "id": "rec66",
        "text": "Session Hijacking is an attack where cybercriminals intercept and alter communications between two parties without their knowledge.",
        "metadata": {
            "section": "Session Hijacking",
            "category": "Network Security"
        }
    },
    {
        "id": "rec67",
        "text": "Business Continuity is a measure used to restore operations and data access following a cybersecurity incident or disaster.",
        "metadata": {
            "section": "Business Continuity",
            "category": "Incident Response & Recovery"
        }
    },
    {
        "id": "rec68",
        "text": "Spear Phishing is an email scam targeting specific individuals or organizations using personalized messages.",
        "metadata": {
            "section": "Spear Phishing",
            "category": "Social Engineering"
        }
    },
    {
        "id": "rec69",
        "text": "White Box Testing is a method that examines a system's internal structure or workings to find vulnerabilities.",
        "metadata": {
            "section": "White Box Testing",
            "category": "Security Testing"
        }
    },
    {
        "id": "rec70",
        "text": "A Supply Chain Attack involves malicious code or scripts inserted into trusted software updates to compromise systems.",
        "metadata": {
            "section": "Supply Chain Attack",
            "category": "Software Security"
        }
    },
    {
        "id": "rec71",
        "text": "Zero Trust Architecture is a cybersecurity model that assumes no user or device should be trusted by default, even inside the network.",
        "metadata": {
            "section": "Zero Trust Architecture",
            "category": "Access Control"
        }
    },
    {
        "id": "rec72",
        "text": "Security Information and Event Management (SIEM) is a system that continuously monitors network traffic to detect and respond to potential security threats.",
        "metadata": {
            "section": "SIEM",
            "category": "Monitoring & Detection"
        }
    },
    {
        "id": "rec73",
        "text": "Data Backup and Recovery is a process that stores encrypted data copies to ensure recoverability in case of a cyberattack.",
        "metadata": {
            "section": "Data Backup and Recovery",
            "category": "Incident Response & Recovery"
        }
    },
    {
        "id": "rec74",
        "text": "Third-Party Risk Management is the practice of evaluating a vendor or partner's security controls before granting them access to systems or data.",
        "metadata": {
            "section": "Third-Party Risk Management",
            "category": "Vendor Security"
        }
    },
    {
        "id": "rec75",
        "text": "Smishing is an attack that uses fraudulent text messages to trick individuals into revealing personal information.",
        "metadata": {
            "section": "Smishing",
            "category": "Social Engineering"
        }
    },
    {
        "id": "rec76",
        "text": "A Red Team Exercise is a security testing technique where ethical hackers simulate real-world attacks to identify weaknesses.",
        "metadata": {
            "section": "Red Team Exercise",
            "category": "Security Testing"
        }
    },
    {
        "id": "rec77",
        "text": "Security Validation is the process of verifying that a system or application meets security requirements before deployment.",
        "metadata": {
            "section": "Security Validation",
            "category": "Security Testing"
        }
    },
    {
        "id": "rec78",
        "text": "An Insider Threat is an individual within an organization who poses a security threat, intentionally or unintentionally.",
        "metadata": {
            "section": "Insider Threat",
            "category": "Human Risk"
        }
    },
    {
        "id": "rec79",
        "text": "An Email Attachment Attack is a malicious attachment or link delivered through email that installs malware on a victim's device.",
        "metadata": {
            "section": "Email Attachment Attack",
            "category": "Malware"
        }
    },
    {
        "id": "rec80",
        "text": "Data Encryption is the process of converting data into an unreadable format to protect it during transmission or storage.",
        "metadata": {
            "section": "Data Encryption",
            "category": "Cryptography"
        }
    },
    {
        "id": "rec81",
        "text": "A Digital Signature uses digital certificates to verify the authenticity and integrity of a message or software.",
        "metadata": {
            "section": "Digital Signature",
            "category": "Cryptography"
        }
    },
    {
        "id": "rec82",
        "text": "Command Injection is a software vulnerability that allows attackers to execute arbitrary commands on a host operating system.",
        "metadata": {
            "section": "Command Injection",
            "category": "Application Security"
        }
    },
    {
        "id": "rec83",
        "text": "Continuous Security Testing is a security practice that involves continuously testing systems to find new vulnerabilities and threats.",
        "metadata": {
            "section": "Continuous Security Testing",
            "category": "Security Testing"
        }
    },
    {
        "id": "rec84",
        "text": "Network Segmentation is a network security measure that separates systems based on trust levels to prevent unauthorized access.",
        "metadata": {
            "section": "Network Segmentation",
            "category": "Network Security"
        }
    },
    {
        "id": "rec85",
        "text": "A Vulnerability Scanner is a tool or platform that automatically scans software for known vulnerabilities and misconfigurations.",
        "metadata": {
            "section": "Vulnerability Scanner",
            "category": "Vulnerability Management"
        }
    },
    {
        "id": "rec86",
        "text": "A Worm is a type of malware that spreads itself across systems without user intervention.",
        "metadata": {
            "section": "Worm",
            "category": "Malware"
        }
    },
    {
        "id": "rec87",
        "text": "A Trojan Horse is a file or program that appears legitimate but hides malicious code.",
        "metadata": {
            "section": "Trojan Horse",
            "category": "Malware"
        }
    },
    {
        "id": "rec88",
        "text": "Configuration Management is a process that ensures systems are configured securely and remain compliant with security policies.",
        "metadata": {
            "section": "Configuration Management",
            "category": "System Management"
        }
    },
    {
        "id": "rec89",
        "text": "Network Access Control is a method of limiting network access to authorized users and devices only.",
        "metadata": {
            "section": "Network Access Control",
            "category": "Access Control"
        }
    },
    {
        "id": "rec90",
        "text": "Defense in Depth is a strategy that combines multiple layers of security controls throughout an IT system.",
        "metadata": {
            "section": "Defense in Depth",
            "category": "Security Architecture"
        }
    },
    {
        "id": "rec91",
        "text": "A Keylogger is software that records user actions, often used maliciously to steal credentials or personal data.",
        "metadata": {
            "section": "Keylogger",
            "category": "Malware"
        }
    },
    {
        "id": "rec92",
        "text": "Spyware is a program that covertly gathers user data without consent and transmits it to a third party.",
        "metadata": {
            "section": "Spyware",
            "category": "Malware"
        }
    },
    {
        "id": "rec93",
        "text": "Identity Verification is the process of confirming that users are who they claim to be using credentials like passwords or biometrics.",
        "metadata": {
            "section": "Identity Verification",
            "category": "Access Control"
        }
    },
    {
        "id": "rec94",
        "text": "Threat Hunting is a proactive cybersecurity approach that focuses on identifying and mitigating threats before they cause harm.",
        "metadata": {
            "section": "Threat Hunting",
            "category": "Threat Intelligence"
        }
    }
]

# Embed and upload to Pinecone
print("Embedding and uploading Infosec policies to Pinecone...")

# Generate embeddings for each policy text
embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [p["text"] for p in policies]
embeddings = embedder.encode(texts, convert_to_numpy=True)

# Build the records for Pinecone upload
records = [
    {
        "id": p["id"],
        "values": embeddings[i].tolist(),
        "metadata": {
            "section": p["metadata"]["section"],
            "category": p["metadata"]["category"],
            "text": p["text"]
        }
    }
    for i, p in enumerate(policies)
]

# Upload to Pinecone
index.upsert(vectors=records, namespace="policies")
print(f"Uploaded {len(records)} hardcoded policies to Pinecone index '{index_name}'.")
