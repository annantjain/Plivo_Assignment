import json
import random
import os

digit_words = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

names = [
    "sandeep kumar", "aarav sharma", "riya verma", "karthik reddy",
    "neha singh", "amit joshi", "vikram patel", "aditya mehra",
    "pooja gupta", "rahul desai", "arjun kapoor"
]

cities = ["mumbai", "delhi", "bangalore", "kolkata", "chennai", "pune"]
locations = ["andheri east", "whitefield", "koramangala", "salt lake", "carter road"]

dates = [
    "fifteenth june twenty twenty one",
    "twenty five july two thousand nineteen",
    "six january nineteen ninety eight",
    "fourth march twenty twenty two",
]

email_patterns = [
    "{name} at gmail dot com",
    "{name} at the rate gmail dot com",
    "{name} gmail com",
    "{name} at yahoo dot co dot in",
]

def digits_to_words(d):
    return " ".join(digit_words[ch] for ch in d)

phone_patterns = [
    "{digits}",
    "{digits_spoken}",
    "plus nine one {digits}",
    "plus nine one {digits_spoken}",
]

cc_patterns = [
    "{digits_spoken}",
    "card number {digits_spoken}",
    "visa card ending with {last4_spoken}",
]

def build_text(template_dict):
    """Create text + entity spans"""
    fields = {}

    # Generate PII fields
    person = random.choice(names)
    fields["PERSON_NAME"] = person

    email_name = person.replace(" ", "")
    email_text = random.choice(email_patterns).format(name=email_name)
    fields["EMAIL"] = email_text

    phone_digits = "".join(random.choice("0123456789") for _ in range(10))
    phone_spoken = digits_to_words(phone_digits)
    phone_text = random.choice(phone_patterns).format(
        digits=phone_digits,
        digits_spoken=phone_spoken
    )
    fields["PHONE"] = phone_text

    cc_digits = "".join(random.choice("0123456789") for _ in range(12))
    cc_spoken = digits_to_words(cc_digits)
    cc_text = random.choice(cc_patterns).format(
        digits_spoken=cc_spoken,
        last4_spoken=digits_to_words(cc_digits[-4:])
    )
    fields["CREDIT_CARD"] = cc_text

    date_text = random.choice(dates)
    fields["DATE"] = date_text

    # Non-PII
    city = random.choice(cities)
    fields["CITY"] = city
    location = random.choice(locations)
    fields["LOCATION"] = location

    # Construct final template
    sentence = template_dict.format(
        PERSON_NAME=fields["PERSON_NAME"],
        EMAIL=fields["EMAIL"],
        PHONE=fields["PHONE"],
        CREDIT_CARD=fields["CREDIT_CARD"],
        DATE=fields["DATE"],
        CITY=fields["CITY"],
        LOCATION=fields["LOCATION"],
    )

    # Compute entity spans
    entities = []
    for label, value in fields.items():
        start = sentence.find(value)
        if start == -1:
            continue
        end = start + len(value)
        entities.append({
            "start": start,
            "end": end,
                            "label": label
        })

    return sentence, entities


templates = [
    "my name is {PERSON_NAME} and my email is {EMAIL} i stay in {CITY} and my phone number is {PHONE}",
    "the person {PERSON_NAME} lives near {LOCATION} the email contact is {EMAIL} and phone is {PHONE}",
    "please note credit card {CREDIT_CARD} and date of booking is {DATE} for customer {PERSON_NAME}",
    "{PERSON_NAME} visited {CITY} on {DATE} and can be contacted at {EMAIL}",
    "payment done using card {CREDIT_CARD} please confirm on {EMAIL}",
]

def generate_jsonl(n_examples, filename):
    with open(filename, "w", encoding="utf8") as f:
        for i in range(n_examples):
            template = random.choice(templates)
            text, ents = build_text(template)
            entry = {
                "id": f"synthetic_{i}",
                "text": text,
                "entities": ents
            }
            f.write(json.dumps(entry) + "\n")

print("Generating synthetic datasets...")

generate_jsonl(700, "data/train_synthetic.jsonl")
generate_jsonl(150, "data/dev_synthetic.jsonl")

print("Done.")
