import json
import random

# Generate synthetic dev data with variations similar to stress test
names = [
    "rahul sharma", "priya patel", "amit kumar", "sneha gupta", "vikram singh",
    "anjali verma", "rajesh malhotra", "kavita joshi", "sanjay mehta", "neha agarwal",
    "arjun reddy", "pooja iyer", "karan chopra", "divya nair", "rohan desai"
]

cities = [
    "mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata",
    "pune", "ahmedabad", "jaipur", "lucknow", "gurgaon", "noida"
]

locations = [
    "koramangala", "indiranagar", "connaught place", "bandra", "andheri",
    "electronic city", "whitefield", "hitech city", "cyber city"
]

# Templates with variations
templates = [
    {
        "text": "this is {name} from {city} my phone is {phone} please call me",
        "entities": lambda d: [
            {"start": d["name_pos"], "end": d["name_pos"] + len(d["name"]), "label": "PERSON_NAME"},
            {"start": d["city_pos"], "end": d["city_pos"] + len(d["city"]), "label": "CITY"},
            {"start": d["phone_pos"], "end": d["phone_pos"] + len(d["phone"]), "label": "PHONE"}
        ]
    },
    {
        "text": "my email is {email} and i live in {location} near {city}",
        "entities": lambda d: [
            {"start": d["email_pos"], "end": d["email_pos"] + len(d["email"]), "label": "EMAIL"},
            {"start": d["location_pos"], "end": d["location_pos"] + len(d["location"]), "label": "LOCATION"},
            {"start": d["city_pos"], "end": d["city_pos"] + len(d["city"]), "label": "CITY"}
        ]
    },
    {
        "text": "haan my naam is {name} and phone number is {phone} we can meet on {date}",
        "entities": lambda d: [
            {"start": d["name_pos"], "end": d["name_pos"] + len(d["name"]), "label": "PERSON_NAME"},
            {"start": d["phone_pos"], "end": d["phone_pos"] + len(d["phone"]), "label": "PHONE"},
            {"start": d["date_pos"], "end": d["date_pos"] + len(d["date"]), "label": "DATE"}
        ]
    },
    {
        "text": "{name} here my card is {card} expires {date} contact me at {email}",
        "entities": lambda d: [
            {"start": d["name_pos"], "end": d["name_pos"] + len(d["name"]), "label": "PERSON_NAME"},
            {"start": d["card_pos"], "end": d["card_pos"] + len(d["card"]), "label": "CREDIT_CARD"},
            {"start": d["date_pos"], "end": d["date_pos"] + len(d["date"]), "label": "DATE"},
            {"start": d["email_pos"], "end": d["email_pos"] + len(d["email"]), "label": "EMAIL"}
        ]
    },
    {
        "text": "call me at {phone} or email {email} my name is {name}",
        "entities": lambda d: [
            {"start": d["phone_pos"], "end": d["phone_pos"] + len(d["phone"]), "label": "PHONE"},
            {"start": d["email_pos"], "end": d["email_pos"] + len(d["email"]), "label": "EMAIL"},
            {"start": d["name_pos"], "end": d["name_pos"] + len(d["name"]), "label": "PERSON_NAME"}
        ]
    }
]

def generate_phone():
    styles = [
        lambda: f"{random.randint(70,99)}{random.randint(10000,99999)} {random.randint(10000,99999)}",
        lambda: " ".join(str(random.randint(6000000000,9999999999))),
        lambda: f"{random.randint(6,9)} {random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)}"
    ]
    return random.choice(styles)()

def generate_email(name):
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "rediffmail.com", "outlook.com"]
    styles = [
        f"{name.replace(' ', '.')}@{random.choice(domains)}",
        f"{name.split()[0]} dot {name.split()[1]} at {random.choice(domains).replace('.', ' dot ')}",
        f"{name.split()[0][0]} dot {name.split()[1]} at {random.choice(domains).replace('.', ' dot ')}"
    ]
    return random.choice(styles)

def generate_card():
    styles = [
        lambda: " ".join([str(random.randint(1000,9999)) for _ in range(4)]),
        lambda: f"{random.randint(1000,9999)}-{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}",
        lambda: " ".join([random.choice(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) for _ in range(16)])[:80]
    ]
    return random.choice(styles)()

def generate_date():
    dates = [
        f"{random.randint(1,28)}/{random.randint(1,12)}/{random.randint(2023,2026)}",
        f"{random.randint(1,28)}-{random.randint(1,12)}-{random.randint(2023,2026)}",
        f"{random.randint(1,28)} {random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])} {random.randint(2023,2026)}"
    ]
    return random.choice(dates)

# Read existing dev data
existing = []
with open("data/dev.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            existing.append(json.loads(line))

print(f"Existing dev samples: {len(existing)}")

# Generate new samples (target ~165 more to reach 175 total)
new_samples = []
for i in range(165):
    template = random.choice(templates)
    name = random.choice(names)
    city = random.choice(cities)
    location = random.choice(locations)
    
    # Build the text with placeholders
    text = template["text"]
    positions = {}
    
    # Replace placeholders and track positions
    if "{name}" in text:
        pos = text.find("{name}")
        text = text.replace("{name}", name, 1)
        positions["name_pos"] = pos
        positions["name"] = name
    
    if "{city}" in text:
        pos = text.find("{city}")
        text = text.replace("{city}", city, 1)
        positions["city_pos"] = pos
        positions["city"] = city
    
    if "{location}" in text:
        pos = text.find("{location}")
        text = text.replace("{location}", location, 1)
        positions["location_pos"] = pos
        positions["location"] = location
    
    if "{phone}" in text:
        phone = generate_phone()
        pos = text.find("{phone}")
        text = text.replace("{phone}", phone, 1)
        positions["phone_pos"] = pos
        positions["phone"] = phone
    
    if "{email}" in text:
        email = generate_email(name)
        pos = text.find("{email}")
        text = text.replace("{email}", email, 1)
        positions["email_pos"] = pos
        positions["email"] = email
    
    if "{card}" in text:
        card = generate_card()
        pos = text.find("{card}")
        text = text.replace("{card}", card, 1)
        positions["card_pos"] = pos
        positions["card"] = card
    
    if "{date}" in text:
        date = generate_date()
        pos = text.find("{date}")
        text = text.replace("{date}", date, 1)
        positions["date_pos"] = pos
        positions["date"] = date
    
    entities = template["entities"](positions)
    
    new_samples.append({
        "id": f"utt_gen_{i+1000}",
        "text": text,
        "entities": entities
    })

# Write augmented dev data
with open("data/dev.jsonl", "w", encoding="utf-8") as f:
    for item in existing:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    for item in new_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Generated {len(new_samples)} new samples")
print(f"Total dev samples now: {len(existing) + len(new_samples)}")
