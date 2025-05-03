#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install python-docx')
get_ipython().system('pip install matplotlib joblib')




import zipfile as zf
files = zf.ZipFile("Separated_CVs_Word.zip", 'r')
files.extractall('CVFo')
files.close()




import os 
path=os.path.abspath('CVFo')
print(path)




import re
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Resume Parser: Reads and extracts data from DOCX files.....
class ResumeParser:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def extract_text_from_docx(self, docx_path):
        try:
            from docx import Document
            doc = Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"[!] Error reading DOCX file: {docx_path}\n{e}")
            return ""

    def parse_resumes(self):
        resumes = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(self.folder_path, filename)
                text = self.extract_text_from_docx(file_path)
                if not text:
                    continue
                # Split resumes based on "Name: " tag
                blocks = text.strip().split("Name: ")
                for block in blocks[1:]:
                    try:
                        # Extract name, job title, and skills using regex
                        name = re.search(r"^(.*?)\n", block).group(1).strip()
                        job_title = re.search(r"Job Title: (.*?)\n", block).group(1).strip()
                        skills_block = re.search(r"Skills: (.+)", block, re.DOTALL).group(1)
                        skills = re.findall(r"([A-Za-z+ ]+)\s\((Basic|Intermediate|Advanced)\)", skills_block)
                        skills = [skill.strip() for skill, _ in skills]
                        resumes.append({"Name": name, "JobTitle": job_title, "Skills": skills})
                    except Exception as e:
                        print(f"[!] Error parsing resume {filename}: {e}")
                        continue
        return resumes

# Model Trainer: Trains a Decision Tree model on resumes ....
class ModelTrainer:
    def __init__(self, resume_data):
        self.df = pd.DataFrame(resume_data)
        self.clf = DecisionTreeClassifier()
        self.mlb = MultiLabelBinarizer()

    def train_and_save(self):
        try:
            # Convert skills to binary format and train the model
            X = self.mlb.fit_transform(self.df['Skills'])
            y = self.df['JobTitle']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.clf.fit(X_train, y_train)

            # Save the trained model and skill encoder
            with open("model.pkl", 'wb') as f:
                pickle.dump(self.clf, f)
            with open("skills_binarizer.pkl", 'wb') as f:
                pickle.dump(self.mlb, f)
            print(" Model trained and saved successfully.")
        except Exception as e:
            print(f"[!] Failed to train or save model: {e}")

# Resume Matcher: Matches resumes with job titles......
class ResumeMatcher:
    def __init__(self, model_path, binarizer_path):
        try:
            # Load the saved model and skill binarizer
            with open(model_path, 'rb') as f:
                self.clf = pickle.load(f)
            with open(binarizer_path, 'rb') as f:
                self.mlb = pickle.load(f)
        except Exception as e:
            print(f"[!] Failed to load model or binarizer: {e}")
            raise

    def calculate_similarity(self, candidate_skills, predicted_job_title):
        # Predefined skills required for each job
        job_skill_map = {
            "Accountant": ["QuickBooks", "Budgeting", "Tax Preparation", "Financial Analysis", "Advanced Excel", "Accounting Software"],
            "Marketing Manager": ["SEO", "Social Media Marketing", "Content Strategy", "Google Analytics", "Market Research", "Brand Management"],
            "Software Engineer": ["Python", "Java", "C++", "SQL", "Web Development", "Machine Learning"],
            "Systems Analyst": ["System Design", "SQL", "Network Security", "Requirements Analysis", "Database Management", "Project Management"],
            "Graphic Designer": ["Photoshop", "Illustrator", "Adobe XD", "Sketch", "Branding", "UI/UX Design"],
            "Human Resources Specialist": ["Payroll", "Recruitment", "Employee Relations", "HR Software", "Labor Law", "Performance Management"]
        }
        required_skills = job_skill_map.get(predicted_job_title, [])
        if not required_skills:
            return 0.0
        matches = sum(1 for skill in candidate_skills if any(skill.lower() == req.lower() for req in required_skills))
        percentage = (matches / len(required_skills)) * 100
        return round(percentage, 2)

    def match_resumes(self, resumes):
        job_skill_map = {
            "Accountant": ["QuickBooks", "Budgeting", "Tax Preparation", "Financial Analysis", "Advanced Excel", "Accounting Software"],
            "Marketing Manager": ["SEO", "Social Media Marketing", "Content Strategy", "Google Analytics", "Market Research", "Brand Management"],
            "Software Engineer": ["Python", "Java", "C++", "SQL", "Web Development", "Machine Learning"],
            "Systems Analyst": ["System Design", "SQL", "Network Security", "Requirements Analysis", "Database Management", "Project Management"],
            "Graphic Designer": ["Photoshop", "Illustrator", "Adobe XD", "Sketch", "Branding", "UI/UX Design"],
            "Human Resources Specialist": ["Payroll", "Recruitment", "Employee Relations", "HR Software", "Labor Law", "Performance Management"]
        }

        results = []

        for res in resumes:
            try:
                # Predict job title from skills
                x = self.mlb.transform([res["Skills"]])
                prediction = self.clf.predict(x)[0]
                match_percentage = self.calculate_similarity(res["Skills"], prediction)

                if match_percentage >= 60:
                    final_job = prediction
                    status = "✔"
                else:
                    # Suggest the closest matching job
                    best_job = None
                    best_score = 0
                    for job, req_skills in job_skill_map.items():
                        matches = sum(1 for skill in res["Skills"] if skill.lower() in map(str.lower, req_skills))
                        percent = (matches / len(req_skills)) * 100
                        if percent > best_score:
                            best_score = percent
                            best_job = job
                    final_job = best_job if best_job else prediction
                    match_percentage = round(best_score, 2)
                    status = "✘"

                results.append({
                    "Name": res["Name"],
                    "Actual Job": res["JobTitle"],
                    "Predicted Job": final_job,
                    "Match %": f"{match_percentage}%",
                    "Correct Match": status
                })
            except Exception as e:
                print(f"[!] Error matching resume ({res.get('Name', 'Unknown')}): {e}")
                continue

        return pd.DataFrame(results)

# --- Main Function ---
def main():
    folder_path = '/home/af9988c7-9b7a-4527-819d-9ae1a8c86c1c/CVFo'  #  folder path:

    # Parse resumes
    parser = ResumeParser(folder_path)
    resumes = parser.parse_resumes()

    # Train model
    trainer = ModelTrainer(resumes)
    trainer.train_and_save()

    # Match resumes and generate results
    matcher = ResumeMatcher("model.pkl", "skills_binarizer.pkl")
    df = matcher.match_resumes(resumes)

    print("\nResume Matching Results:\n")
    print(df.to_string(index=False))

    # --- Visualization: Pie Chart ---
    matched = df["Correct Match"].value_counts().get("✔", 0)
    total = len(df)
    plt.figure(figsize=(5, 5))
    plt.pie([matched, total - matched], labels=["Matched", "Not Matched"],
            autopct="%1.1f%%", colors=["green", "red"], startangle=90)
    plt.title("Matching Accuracy")
    plt.show()

    # --- Visualization: Bar Chart ---
    plt.figure(figsize=(10, 6))
    df["Predicted Job"].value_counts().plot(kind="bar", color="pink")
    plt.title("Predicted Job Titles Distribution")
    plt.xlabel("Job Title")
    plt.ylabel("Number of Candidates")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()




import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model and binarizer
with open("model.pkl", 'rb') as f:
    clf = pickle.load(f)
with open("skills_binarizer.pkl", 'rb') as f:
    mlb = pickle.load(f)

# Convert to lists for plotting
feature_names = mlb.classes_.tolist()
class_names = clf.classes_.tolist()

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()




import pickle

# Class to test a single resume using user input
class ResumeTester:
    def __init__(self, model_path, binarizer_path):
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(binarizer_path, 'rb') as f:
            self.mlb = pickle.load(f)
        self.known_skills = set(self.mlb.classes_)

    def predict_job(self, skills):
        # Validate skills
        invalid_skills = [s for s in skills if s not in self.known_skills]
        if invalid_skills:
            raise ValueError(f"Unrecognized skills: {', '.join(invalid_skills)}.\n"
                             f"Sorry, no job match found for your skills.")

        # Predict job
        x = self.mlb.transform([skills])
        return self.clf.predict(x)[0]

# Get user input
def get_user_input():
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    skills_input = input("Enter your skills (comma-separated): ")
    skills = [s.strip() for s in skills_input.split(",")]
    return name, email, skills

# Instantiating the ResumeTester class and processing input
tester = ResumeTester("model.pkl", "skills_binarizer.pkl")
name, email, skills = get_user_input()

try:
    predicted_job = tester.predict_job(skills)
    print(f"\n{name}, based on your skills, the suggested job is: {predicted_job}")
except ValueError as e:
    print(f"\n[ERROR] {e}") 






