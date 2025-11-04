print("hello")


# Import the Pinecone library
from pinecone import Pinecone

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_6e2JzS_PZYtxdavrAwT6JVaBzFVB2gNs4j6DVeu8LQwNf8WfoT3bwmrbQSCWpHHBvWh91P")

# Create a dense index with integrated embedding
index_name = "developer-quickstart-py"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

index_name = "developer-quickstart-py"



records = [{ "_id": "rec1", "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history" },
           { "_id": "rec2", "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science" },
            { "_id": "rec3", "chunk_text": "Albert Einstein developed the theory of relativity.", "category": "science" },
            { "_id": "rec4", "chunk_text": "The mitochondrion is often called the powerhouse of the cell.", "category": "biology" },
            { "_id": "rec5", "chunk_text": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature" },
            { "_id": "rec6", "chunk_text": "Water boils at 100C under standard atmospheric pressure.", "category": "physics" },
            { "_id": "rec7", "chunk_text": "The Great Wall of China was built to protect against invasions.", "category": "history" },
            { "_id": "rec8", "chunk_text": "Honey never spoils due to its low moisture content and acidity.", "category": "food science" },
            { "_id": "rec9", "chunk_text": "The speed of light in a vacuum is approximately 299,792 km/s.", "category": "physics" },
            { "_id": "rec10", "chunk_text": "Newton's laws describe the motion of objects.", "category": "physics" },
            { "_id": "rec11", "chunk_text": "The human brain has approximately 86 billion neurons.", "category": "biology" },
            { "_id": "rec12", "chunk_text": "The Amazon Rainforest is one of the most biodiverse places on Earth.", "category": "geography" },
            { "_id": "rec13", "chunk_text": "Black holes have gravitational fields so strong that not even light can escape.", "category": "astronomy" },
            { "_id": "rec14", "chunk_text": "The periodic table organizes elements based on their atomic number.", "category": "chemistry" },
            { "_id": "rec15", "chunk_text": "Leonardo da Vinci painted the Mona Lisa.", "category": "art" },
            { "_id": "rec16", "chunk_text": "The internet revolutionized communication and information sharing.", "category": "technology" },
            { "_id": "rec17", "chunk_text": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history" },
            { "_id": "rec18", "chunk_text": "Dogs have an incredible sense of smell, much stronger than humans.", "category": "biology" },
            { "_id": "rec19", "chunk_text": "The Pacific Ocean is the largest and deepest ocean on Earth.", "category": "geography" },
            { "_id": "rec20", "chunk_text": "Chess is a strategic game that originated in India.", "category": "games" },
            { "_id": "rec21", "chunk_text": "The Statue of Liberty was a gift from France to the United States.", "category": "history" },
            { "_id": "rec22", "chunk_text": "Coffee contains caffeine, a natural stimulant.", "category": "food science" },
            { "_id": "rec23", "chunk_text": "Thomas Edison invented the practical electric light bulb.", "category": "inventions" },
            { "_id": "rec24", "chunk_text": "The moon influences ocean tides due to gravitational pull.", "category": "astronomy" },
            { "_id": "rec25", "chunk_text": "DNA carries genetic information for all living organisms.", "category": "biology" },
            { "_id": "rec26", "chunk_text": "Rome was once the center of a vast empire.", "category": "history" },
            { "_id": "rec27", "chunk_text": "The Wright brothers pioneered human flight in 1903.", "category": "inventions" },
            { "_id": "rec28", "chunk_text": "Bananas are a good source of potassium.", "category": "nutrition" },
            { "_id": "rec29", "chunk_text": "The stock market fluctuates based on supply and demand.", "category": "economics" },
            { "_id": "rec30", "chunk_text": "A compass needle points toward the magnetic north pole.", "category": "navigation" },
            { "_id": "rec31", "chunk_text": "The universe is expanding, according to the Big Bang theory.", "category": "astronomy" },
            { "_id": "rec32", "chunk_text": "Elephants have excellent memory and strong social bonds.", "category": "biology" },
            { "_id": "rec33", "chunk_text": "The violin is a string instrument commonly used in orchestras.", "category": "music" },
            { "_id": "rec34", "chunk_text": "The heart pumps blood throughout the human body.", "category": "biology" },
            { "_id": "rec35", "chunk_text": "Ice cream melts when exposed to heat.", "category": "food science" },
            { "_id": "rec36", "chunk_text": "Solar panels convert sunlight into electricity.", "category": "technology" },
            { "_id": "rec37", "chunk_text": "The French Revolution began in 1789.", "category": "history" },
            { "_id": "rec38", "chunk_text": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history" },
            { "_id": "rec39", "chunk_text": "Rainbows are caused by light refracting through water droplets.", "category": "physics" },
            { "_id": "rec40", "chunk_text": "Mount Everest is the tallest mountain in the world.", "category": "geography" },
            { "_id": "rec41", "chunk_text": "Octopuses are highly intelligent marine creatures.", "category": "biology" },
            { "_id": "rec42", "chunk_text": "The speed of sound is around 343 meters per second in air.", "category": "physics" },
            { "_id": "rec43", "chunk_text": "Gravity keeps planets in orbit around the sun.", "category": "astronomy" },
            { "_id": "rec44", "chunk_text": "The Mediterranean diet is considered one of the healthiest in the world.", "category": "nutrition" },
            { "_id": "rec45", "chunk_text": "A haiku is a traditional Japanese poem with a 5-7-5 syllable structure.", "category": "literature" },
            { "_id": "rec46", "chunk_text": "The human body is made up of about 60% water.", "category": "biology" },
            { "_id": "rec47", "chunk_text": "The Industrial Revolution transformed manufacturing and transportation.", "category": "history" },
            { "_id": "rec48", "chunk_text": "Vincent van Gogh painted Starry Night.", "category": "art" },
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
            { "_id": "rec96", "chunk_text": "Maintain a cybersecurity safeguard validation (audit) plan that documents the assessments the organization shall perform to validate the quality of the organization's cybersecurity safeguards. Ensure that the organization's cybersecurity safeguard validation (audit)plan is a multi-year plan that regularly addresses all of the scopes the organization should assess. Ensure that the organization's cybersecurity safeguard validation (audit)plan establishes criticality rankings for each of the assessment scopes in its assessment plan. Ensure that the organization's cybersecurity safeguard validation (audit)plan defines who should perform each assessment scope in its assessment plan. Ensure that the organization's cybersecurity safeguard validation (audit)plan includes each of the cybersecurity penetration testing scopes it should assess regularly. Ensure that the organization's cybersecurity safeguard validation (audit)plan includes software application penetration tests in its assessment plan. Ensure that the organization's cybersecurity safeguard validation (audit)plan includes red team cybersecurity assessments in its assessment plan.Ensure that the organization's cybersecurity safeguard validation (audit)plan defines where cybersecurity penetration testing should be performed only against test systems due to the sensitivity of such systems. Ensure that the organization's cybersecurity safeguard validation (audit) plan defines how cybersecurity penetration testing should utilize vulnerability scanners as a part of the assessments.", "category": "phishing" }]

    

    # Target the index
dense_index = pc.Index(index_name)

# Upsert the records into a namespace
dense_index.upsert_records("testing-namespace", records)




# Wait for the upserted vectors to be indexed
import time
time.sleep(10)

# View stats for the index
stats = dense_index.describe_index_stats()
print(stats)


# Define the query
query = "Europe"

# Search the dense index
results = dense_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    }
)

# Print the results
for hit in results['result']['hits']:
        print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")


