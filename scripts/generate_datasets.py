import os
import csv
import random
from faker import Faker

# Set up Faker with Indian locale
fake = Faker('en_IN')
Faker.seed(42)
random.seed(42)

# Generate synthetic PII values
def generate_aadhaar():
    # 12 digits formatted as XXXX-XXXX-XXXX
    return f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

def generate_pan():
    # 5 letters, 4 digits, 1 letter
    letters1 = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    digits = "".join(random.choices("0123456789", k=4))
    letter2 = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{letters1}{digits}{letter2}"

def generate_voter_id():
    # 3 letters, 7 digits
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    digits = "".join(random.choices("0123456789", k=7))
    return f"{letters}{digits}"

def generate_driving_license():
    # MH1220181234567 or DL-1420220123456
    state = random.choice(["MH", "DL", "KA", "TN", "UP", "HR"])
    code = f"{random.randint(10, 99)}"
    year = f"{random.randint(2010, 2026)}"
    num = "".join(random.choices("0123456789", k=7))
    return f"{state}{code}{year}{num}"

def generate_ifsc(bank=None):
    if not bank:
        bank = random.choice(["SBIN", "HDFC", "ICIC", "BARB", "PNJB"])
    num = "".join(random.choices("0123456789", k=6))
    return f"{bank}0{num}"

def generate_upi():
    user = fake.user_name()
    handle = random.choice(["upi", "ybl", "okaxis", "paytm", "okhdfcbank", "icici"])
    return f"{user}@{handle}"

def generate_gstin(pan_code=None):
    state_code = f"{random.randint(1, 37):02d}"
    if not pan_code:
        pan_code = generate_pan()
    entity_code = random.choice("12")
    check_digit = random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{state_code}{pan_code}{entity_code}Z{check_digit}"

def generate_medical_license():
    return f"MCI-{random.randint(10000, 99999)}"

def generate_vehicle_number():
    state = random.choice(["MH", "DL", "KA", "TN", "UP", "HR"])
    district = f"{random.randint(1, 99):02d}"
    series = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    num = f"{random.randint(1000, 9999)}"
    return f"{state}-{district}-{series}-{num}"

def generate_passport():
    # Z followed by 7 digits
    letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(random.choices("0123456789", k=7))
    return f"{letter}{digits}"

def generate_ip():
    return f"{random.randint(1, 254)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def generate_synthetic_dataset(filename, num_rows=100):
    headers = [
        "Person", "Organization", "Location", "Email", "Phone_Number", 
        "Date_Time", "URL", "IP_Address", "Credit_Card", "Bank_Account", 
        "Passport", "Drivers_License", "National_ID", "Medical_License", 
        "Vehicle_Number", "Age", "Username", "Aadhaar", "PAN", 
        "Voter_ID", "IFSC_Code", "UPI_ID", "GSTIN", "Mobile_IN"
    ]
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for _ in range(num_rows):
            pan = generate_pan()
            bank = random.choice(["SBIN", "HDFC", "ICIC"])
            row = [
                fake.name(),
                fake.company(),
                fake.city(),
                fake.email(),
                fake.phone_number(),
                fake.date_time().isoformat(),
                fake.url(),
                generate_ip(),
                fake.credit_card_number(),
                fake.bban(),
                generate_passport(),
                generate_driving_license(),
                f"NID-{random.randint(100000, 999999)}",
                generate_medical_license(),
                generate_vehicle_number(),
                random.randint(18, 90),
                fake.user_name(),
                generate_aadhaar(),
                pan,
                generate_voter_id(),
                generate_ifsc(bank),
                generate_upi(),
                generate_gstin(pan),
                f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
            ]
            writer.writerow(row)
    print(f"Generated synthetic dataset: {filename}")

def generate_simulated_kaggle_dataset(filename, num_rows=50):
    # Simulated Kaggle Customer Support / Feedback Dataset with Indian Context containing PII
    headers = ["ticket_id", "category", "customer_query", "labeled_pii"]
    
    scenarios = [
        {
            "category": "Refund / Billing",
            "template": "Hi, I am {name} from {city}. I paid for the invoice through UPI but the money got deducted and the order was not processed. My UPI ID is {upi} and I also have my bank account details: Account Number {acc_num}, IFSC {ifsc}. Please issue the refund to this bank account as soon as possible. My PAN is {pan}.",
            "pii_keys": ["name", "city", "upi", "acc_num", "ifsc", "pan"]
        },
        {
            "category": "Verification Failure",
            "template": "Respected Officer, my Aadhaar card verification failed for my connection. My Aadhaar number is {aadhaar} and my name is {name}. Can I submit my Voter ID {voter_id} or Driving License {dl} instead? Please contact me at {phone} or {email}.",
            "pii_keys": ["aadhaar", "name", "voter_id", "dl", "phone", "email"]
        },
        {
            "category": "Corporate Registration",
            "template": "Hello, we need to register our enterprise {company} on the billing portal. Our GSTIN is {gstin} and our billing address is in {city}. The point of contact is {name} ({email}, phone: {phone}). The vehicle registration for delivery is {vehicle_num}.",
            "pii_keys": ["company", "gstin", "city", "name", "email", "phone", "vehicle_num"]
        },
        {
            "category": "Account Security",
            "template": "My account was compromised. Someone changed my username from {username} to something else. They also attempted a transaction using my credit card ending in {cc}. I am currently at IP {ip}. Help! My age is {age} and my medical license is {med_license}.",
            "pii_keys": ["username", "cc", "ip", "age", "med_license"]
        },
        {
            "category": "Travel / Passport Verification",
            "template": "Dear Team, I am submitting my passport copy for booking confirmation. Passport Number is {passport}. Name: {name}. Born on {datetime}. I can be reached at {email} or {phone}.",
            "pii_keys": ["passport", "name", "datetime", "email", "phone"]
        }
    ]
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(1, num_rows + 1):
            scenario = random.choice(scenarios)
            # Generate random field values
            pan = generate_pan()
            bank = random.choice(["SBIN", "HDFC", "ICIC"])
            
            vals = {
                "name": fake.name(),
                "city": fake.city(),
                "company": fake.company(),
                "email": fake.email(),
                "phone": f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}",
                "username": fake.user_name(),
                "cc": fake.credit_card_number(),
                "ip": generate_ip(),
                "datetime": fake.date_time().isoformat(),
                "age": str(random.randint(18, 75)),
                "acc_num": fake.bban(),
                "ifsc": generate_ifsc(bank),
                "upi": generate_upi(),
                "pan": pan,
                "gstin": generate_gstin(pan),
                "aadhaar": generate_aadhaar(),
                "voter_id": generate_voter_id(),
                "dl": generate_driving_license(),
                "passport": generate_passport(),
                "vehicle_num": generate_vehicle_number(),
                "med_license": generate_medical_license()
            }
            
            query = scenario["template"].format(**vals)
            labeled_pii = {k: vals[k] for k in scenario["pii_keys"]}
            
            writer.writerow([
                f"TKT-{i:04d}",
                scenario["category"],
                query,
                str(labeled_pii)
            ])
    print(f"Generated simulated Kaggle dataset: {filename}")

if __name__ == "__main__":
    generate_synthetic_dataset("dataset/synthetic_pii.csv", 100)
    generate_simulated_kaggle_dataset("dataset/kaggle_pii.csv", 50)
