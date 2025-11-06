from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# === Initialize Pinecone ===
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

# === Hardcoded cybersecurity policies ===
policies = [
            { "_id": "rec49", "chunk_text": "Strong passwords are long, the more characters a password has the stronger it is. We recommend a minimum of 16 characters in all work related passwords. In addition, we encourage the use of passphrases, passwords made up of multiple words. Examples include 'It's time for vacation' or 'block-curious-sunny-leaves'. Passphrases are both easy to remember and type yet meet the strength requirements. ", "category": "password" },
            { "_id": "rec50", "chunk_text": "Password cracking or guessing may be performed on a periodic or random basis by the Infosec Team or its delegates. If a password is guessed or cracked during one of these scans, the user will be required to change. ", "category": "password" },
            { "_id": "rec51", "chunk_text": "Our Network Device Management Policy aims to establish a comprehensive framework for the secure configuration, monitoring, and management of network devices within our organization. This policy aims to provide clear guidelines and procedures for properly administering, maintaining, and protecting network devices, including routers, switches, firewalls, and wireless access points. This policy seeks to minimize the risk of unauthorized access, data breaches, and network disruptions by implementing effective network device management practices. By enforcing secure configurations, regular patch management, and proactive monitoring, we strive to ensure our network infrastructure's availability, integrity, and confidentiality, protect against emerging threats, and maintain compliance with industry standards and regulatory requirements. By prioritizing network device management, we strengthen our overall cybersecurity posture, optimize network performance, and maintain the trust and confidence of our stakeholders.", "category": "network device management" },
            { "_id": "rec52", "chunk_text": "NDM-01 Maintain an inventory of each of the organization's approved network devices.NDM-02 Maintain network device cybersecurity configuration benchmarks for the organization's authorized network devices.NDM-03 Ensure the organization's network devices are managed from an approved, dedicated management network subnet.NDM-04 Ensure the organization's network devices are not managed from a remote (or Internet-based) network.NDM-05 Ensure the organization's network devices are managed from an approved Privileged Account Management (PAM) system or management jump box.NDM-06 Ensure the organization's network devices require using Multi-Factor Authentication (MFA) to access the device.NDM-07 Ensure the organization's network devices use encrypted remote management protocols (such as SSH or TLS).NDM-08 Maintain a network device management system to manage each organization's approved network device.NDM-09 Ensure the organization's network device management system regularly scans for new network devices to add to the organization's network device inventory.NDM-10 Ensure the organization's network device management system monitors each network device's status and logs and alerts when it is offline.NDM-11 Ensure the organization's network device management system performs IP Address Management (IPAM) for each network (including DHCP scopes).", "category": "network device management" },
            { "_id": "rec53", "chunk_text": "Our Data Privacy Policy aims to establish a comprehensive framework for protecting the privacy and confidentiality of personal and sensitive data entrusted to our organization. This policy aims to provide clear guidelines and procedures for data collection, storage, use, disclosure, and disposal in compliance with applicable privacy laws, regulations, and industry best practices. By implementing effective data privacy practices, this policy seeks to ensure the lawful and ethical handling of personal information, safeguard the rights and privacy of individuals, and maintain the trust and confidence of our customers, partners, and stakeholders. Through robust data protection measures, privacy impact assessments, and ongoing monitoring, we strive to mitigate the risks of unauthorized access, data breaches, and misuse of personal information while fostering transparency, accountability, and compliance in our data handling practices.", "category": "privacy" },
            { "_id": "rec54", "chunk_text": "PRV-01 Maintain a transparent, documented privacy program that documents the organization's safeguards to address data privacy.PRV-02 Ensure that the organization's documented privacy program defines a process for performing data processing authorizations (authorizing, maintaining, and revoking).PRV-03 Ensure that the organization's documented privacy program defines a process for reviewing, transferring, disclosing, modifying, or deleting data from the organization's information systems for privacy purposes.PRV-04 Ensure that the organization's documented privacy program defines a process for recording and maintaining an individual's privacy preferences.PRV-05 Ensure that the organization's documented privacy program defines a process for recording, maintaining, and reviewing stakeholder goals for data privacy.PRV-06 Ensure that the organization's documented privacy program defines a process for evaluating the organization's use of data for bias.PRV-07 Ensure that the organization's documented privacy program defines a process for recording and evaluating data provenance and lineage.PRV-08 Ensure that the organization's documented privacy program defines a process for limiting the identification or inference of individuals when processing data.PRV-09 Ensure that the organization's documented privacy program defines a process for replacing attribute values with attribute references in the organization's information systems for privacy purposes. PRV-10 Ensure that the organization's documented privacy program defines a process for informing customers and external business partners about how their data is being used and the organization's privacy goals.PRV-11 Ensure that the organization's documented privacy program defines a process to obtain feedback from individuals regarding the organization's use of data and the associated privacy risks.PRV-12 Ensure that the organization's documented privacy program defines a process to allow individuals to request data corrections to their data.PRV-13 Ensure that the organization's documented privacy program defines a process to allow individuals to request data deletions of their data (right to be forgotten).PRV-14 Ensure that the organization's documented privacy program defines a process for sharing only appropriate data with third parties.PRV-15 Maintain a technology platform to record the organization's efforts related to its data privacy program.PRV-16 Ensure the organization's privacy record system tracks individuals' stated privacy preferences.PRV-17 Ensure that the organization's privacy record system tracks data correction and deletion requests and the organization's response.PRV-18 Ensure the organization's privacy record system tracks data disclosures or sharing personal information with third-parties.", "category": "privacy" },
            { "_id": "rec55", "chunk_text": "SQL Injection is a type of attack where malicious SQL statements are inserted into an entry field for execution.", "category": "sql_injection" },
            { "_id": "rec56", "chunk_text": "An Intrusion Detection System (IDS) is a system that detects and alerts on potential security breaches or malicious activities within a network.", "category": "intrusion_detection_system" },
            { "_id": "rec57", "chunk_text": "An Intrusion Prevention System (IPS) is designed to automatically block or prevent detected intrusions from succeeding.", "category": "intrusion_prevention_system" },
            { "_id": "rec58", "chunk_text": "Security Governance is a framework that outlines how an organization manages and governs cybersecurity risk.", "category": "security_governance" },
            { "_id": "rec59", "chunk_text": "Data Exfiltration is the unauthorized copying, transfer, or use of sensitive data outside an organization's boundaries.", "category": "data_exfiltration" },
            { "_id": "rec60", "chunk_text": "Spoofing is a fake website or service designed to trick users into revealing sensitive information.", "category": "spoofing" },
            { "_id": "rec61", "chunk_text": "A Zero-Day Attack targets software vulnerabilities before developers have released a patch or fix.", "category": "zero_day_attack" },
            { "_id": "rec62", "chunk_text": "Confidentiality is the process of ensuring that data is accessible only to those authorized to view it.", "category": "confidentiality" },
            { "_id": "rec63", "chunk_text": "Cloud Security is the discipline focused on protecting cloud-based data, applications, and infrastructure.", "category": "cloud_security" },
            { "_id": "rec64", "chunk_text": "An Information Security Policy is a document that defines an organization's security practices, roles, and responsibilities.", "category": "information_security_policy" },
            { "_id": "rec65", "chunk_text": "A Security Patch is a software update designed to fix vulnerabilities or bugs in applications and systems.", "category": "security_patch" },
            { "_id": "rec66", "chunk_text": "Session Hijacking is an attack where cybercriminals intercept and alter communications between two parties without their knowledge.", "category": "session_hijacking" },
            { "_id": "rec67", "chunk_text": "Business Continuity is a measure used to restore operations and data access following a cybersecurity incident or disaster.", "category": "business_continuity" },
            { "_id": "rec68", "chunk_text": "Spear Phishing is an email scam targeting specific individuals or organizations using personalized messages.", "category": "spear_phishing" },
            { "_id": "rec69", "chunk_text": "White Box Testing is a method that examines a system's internal structure or workings to find vulnerabilities.", "category": "white_box_testing" },
            { "_id": "rec70", "chunk_text": "A Supply Chain Attack involves malicious code or scripts inserted into trusted software updates to compromise systems.", "category": "supply_chain_attack" },
            { "_id": "rec71", "chunk_text": "Zero Trust Architecture is a cybersecurity model that assumes no user or device should be trusted by default, even inside the network.", "category": "zero_trust_architecture" },
            { "_id": "rec72", "chunk_text": "Security Information and Event Management (SIEM) is a system that continuously monitors network traffic to detect and respond to potential security threats.", "category": "security_information_and_event_management" },
            { "_id": "rec73", "chunk_text": "Data Backup and Recovery is a process that stores encrypted data copies to ensure recoverability in case of a cyberattack.", "category": "data_backup_and_recovery" },
            { "_id": "rec74", "chunk_text": "Third-Party Risk Management is the practice of evaluating a vendor or partner's security controls before granting them access to systems or data.", "category": "third_party_risk_management" },
            { "_id": "rec75", "chunk_text": "Smishing is an attack that uses fraudulent text messages to trick individuals into revealing personal information.", "category": "smishing" },
            { "_id": "rec76", "chunk_text": "A Red Team Exercise is a security testing technique where ethical hackers simulate real-world attacks to identify weaknesses.", "category": "red_team_exercise" },
            { "_id": "rec77", "chunk_text": "Security Validation is the process of verifying that a system or application meets security requirements before deployment.", "category": "security_validation" },
            { "_id": "rec78", "chunk_text": "An Insider Threat is an individual within an organization who poses a security threat, intentionally or unintentionally.", "category": "insider_threat" },
            { "_id": "rec79", "chunk_text": "An Email Attachment Attack is a malicious attachment or link delivered through email that installs malware on a victim's device.", "category": "email_attachment_attack" },
            { "_id": "rec80", "chunk_text": "Data Encryption is the process of converting data into an unreadable format to protect it during transmission or storage.", "category": "data_encryption" },
            { "_id": "rec81", "chunk_text": "A Digital Signature uses digital certificates to verify the authenticity and integrity of a message or software.", "category": "digital_signature" },
            { "_id": "rec82", "chunk_text": "Command Injection is a software vulnerability that allows attackers to execute arbitrary commands on a host operating system.", "category": "command_injection" },
            { "_id": "rec83", "chunk_text": "Continuous Security Testing is a security practice that involves continuously testing systems to find new vulnerabilities and threats.", "category": "continuous_security_testing" },
            { "_id": "rec84", "chunk_text": "Network Segmentation is a network security measure that separates systems based on trust levels to prevent unauthorized access.", "category": "network_segmentation" },
            { "_id": "rec85", "chunk_text": "A Vulnerability Scanner is a tool or platform that automatically scans software for known vulnerabilities and misconfigurations.", "category": "vulnerability_scanner" },
            { "_id": "rec86", "chunk_text": "A Worm is a type of malware that spreads itself across systems without user intervention.", "category": "worm" },
            { "_id": "rec87", "chunk_text": "A Trojan Horse is a file or program that appears legitimate but hides malicious code.", "category": "trojan_horse" },
            { "_id": "rec88", "chunk_text": "Configuration Management is a process that ensures systems are configured securely and remain compliant with security policies.", "category": "configuration_management" },
            { "_id": "rec89", "chunk_text": "Network Access Control is a method of limiting network access to authorized users and devices only.", "category": "network_access_control" },
            { "_id": "rec90", "chunk_text": "Defense in Depth is a strategy that combines multiple layers of security controls throughout an IT system.", "category": "defense_in_depth" },
            { "_id": "rec91", "chunk_text": "A Keylogger is software that records user actions, often used maliciously to steal credentials or personal data.", "category": "keylogger" },
            { "_id": "rec92", "chunk_text": "Spyware is a program that covertly gathers user data without consent and transmits it to a third party.", "category": "spyware" },
            { "_id": "rec93", "chunk_text": "Identity Verification is the process of confirming that users are who they claim to be using credentials like passwords or biometrics.", "category": "identity_verification" },
            { "_id": "rec94", "chunk_text": "Threat Hunting is a proactive cybersecurity approach that focuses on identifying and mitigating threats before they cause harm.", "category": "threat_hunting" },
            { "_id": "rec95", "chunk_text": "Physics explains how airplanes fly due to the principles of lift and aerodynamics.", "category": "physics" },
]

# === Embed and upload to Pinecone ===
print("Embedding and uploading Infosec policies to Pinecone...")
embeddings = embedder.encode([p["text"] for p in policies], convert_to_numpy=True)

records = [
    {
        "id": f"policy_{i}",
        "values": embeddings[i].tolist(),
        "metadata": {"section": p["section"], "text": p["text"]}
    }
    for i, p in enumerate(policies)
]

index.upsert(vectors=records, namespace="policies")

print(f" Uploaded {len(policies)} hardcoded policies to Pinecone index '{index_name}'.")

