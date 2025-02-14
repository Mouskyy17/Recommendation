# Importations principales
import streamlit as st
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import openai
from PyPDF2 import PdfReader
import tempfile
import os

# Configuration initiale
st.set_page_config(page_title="World Like Home - Assistant Mobilité", layout="wide")
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
    nlp = spacy.load("fr_core_news_md")

# ---- Partie Backend ----
class StudentAnalyzer:
    def __init__(self):
        self.riasec_mapping = {
            'R': 'Réaliste', 'I': 'Investigateur', 'A': 'Artistique',
            'S': 'Social', 'E': 'Entreprenant', 'C': 'Conventionnel'
        }
    
    def parse_cv(self, pdf_file):
        """Extrait le texte d'un CV PDF"""
        text = ""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.getvalue())
            reader = PdfReader(tmp.name)
            for page in reader.pages:
                text += page.extract_text()
        os.unlink(tmp.name)
        return text


    # Modifiez la méthode _process_grades
def _process_grades(self, pdf_file):
    """Traite les relevés de notes PDF avec extraction intelligente"""
    if pdf_file is None:
        return {}

    try:
        # Sauvegarde temporaire du PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name

        # Extraction des tableaux avec Camelot
        tables = camelot.read_pdf(tmp_path, flavor='stream', pages='all')
        
        # Recherche des notes dans tout le texte
        text = self.parse_cv(pdf_file)  # Utilise la méthode existante d'extraction texte
        grades = self._find_grades_in_text(text)
        
        # Nettoyage du fichier temporaire
        os.unlink(tmp_path)
        
        return {
            'moyenne_generale': sum(grades.values())/len(grades) if grades else 0,
            'matieres_fortes': sorted(grades, key=grades.get, reverse=True)[:3],
            'matieres_faibles': sorted(grades, key=grades.get)[:3],
            'raw_data': grades
        }
        
    except Exception as e:
        st.error(f"Erreur d'analyse du bulletin : {str(e)}")
        return {}

def _find_grades_in_text(self, text):
    """Détection des notes par expressions régulières"""
    grade_pattern = r"""
        (\b(?:MATIERE|UE|COURS)\b[\s\S]*?)?  # Nom matière
        (\b[A-ZÉÈÀÊÇ\s-]+\b)                # Nom matière alternative
        [\s\S]*?
        (?:Note|NOTE|Appréciation)\D*        # Libellé note
        (\d{1,2}[.,]\d{2})                  # Format note
    """
    
    matches = re.findall(grade_pattern, text, re.VERBOSE | re.IGNORECASE)
    grades = {}
    
    for match in matches:
        matiere = match[0].strip() if match[0] else match[1].strip()
        note = float(match[2].replace(',', '.'))
        grades[matiere] = note
        
    return grades

    def _analyze_sentiment(self, text):
        """Analyse simplifiée des aspirations (exemple)"""
        doc = nlp(text)
        return {
            'mots_cles': [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB']],
            'sentiment': 'positif' if sum(token.sentiment for token in doc) > 0 else 'neutre/negatif'
        }


    def analyze_profile(self, cv_text, grades, aspirations):
        """Analyse NLP du profil étudiant"""
        doc = nlp(cv_text + " " + aspirations)
        skills = [ent.text for ent in doc.ents if ent.label_ in ['SKILL', 'DOMAIN']]
        return {
            'competences': list(set(skills)),
            'notes': self._process_grades(grades),
            'aspirations': self._analyze_sentiment(aspirations)
        }

class EstablishmentMatcher:
    def __init__(self):
        self.etablissements = self._load_etablissements_data()
    
    def _load_etablissements_data(self):
        """Charge les données des établissements depuis l'API Parcoursup"""
        response = requests.get("https://mesr.opendatasoft.com/api/records/1.0/search/?dataset=fr-esr-parcoursup")
        return pd.DataFrame([r['fields'] for r in response.json()['records']])

    def match_establishments(self, student_profile):
        """Algorithme de matching des établissements"""
        # Logique de matching (exemple simplifié)
        return self.etablissements.sample(7).to_dict('records')

# ---- Partie Frontend ----
def main():
    st.title("🌍 World Like Home - Assistant de Mobilité Étudiante")
    
    # Sidebar - Upload des documents
    with st.sidebar:
        st.header("📤 Téléchargement des documents")
        cv_file = st.file_uploader("CV (PDF)", type=["pdf"])
        grades_file = st.file_uploader("Relevés de notes (Pdf)", type=["pdf"])
        aspirations = st.text_area("Aspirations professionnelles")
        
        if st.button("Lancer l'analyse"):
            with st.spinner("Analyse en cours..."):
                # Traitement des données
                analyzer = StudentAnalyzer()
                cv_text = analyzer.parse_cv(cv_file)
                student_profile = analyzer.analyze_profile(cv_text, grades_file, aspirations)
                
                # Matching des établissements
                matcher = EstablishmentMatcher()
                recommendations = matcher.match_establishments(student_profile)
                
                # Stockage en session
                st.session_state['recommendations'] = recommendations
                st.session_state['profile'] = student_profile

    # Affichage des résultats
    if 'recommendations' in st.session_state:
        st.header("🔎 Recommandations personnalisées")
        
        # Section Profil
        with st.expander("Profil étudiant analysé"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Compétences clés")
                st.write(", ".join(st.session_state['profile']['competences'][:10]))
                
            with col2:
                st.subheader("Aspirations analysées")
                st.write(st.session_state['profile']['aspirations'])

        # Section Établissements
        st.subheader("🎓 Établissements recommandés")
        for etab in st.session_state['recommendations']:
            with st.expander(f"{etab['etablissement']} - {etab['ville']}"):
                col1, col2, col3 = st.columns([1,2,1])
                with col1:
                    st.image(etab.get('logo', 'https://via.placeholder.com/150'), width=100)
                with col2:
                    st.write(f"**Programme:** {etab['formation']}")
                    st.write(f"**Spécialités:** {etab['specialites']}")
                with col3:
                    if st.button("Générer lettre", key=etab['etablissement']):
                        generate_cover_letter(etab)

def generate_cover_letter(etablissement):
    """Génère une lettre de motivation avec GPT"""
    prompt = f"""
    Rédige une lettre de motivation pour {etablissement['formation']} à {etablissement['etablissement']}.
    Inclure les éléments suivants:
    - Compétences: {st.session_state['profile']['competences']}
    - Aspirations: {st.session_state['profile']['aspirations']}
    Ton style: professionnel et motivé
    """
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    
    with st.expander("Lettre générée"):
        st.write(response.choices[0].text)
        st.download_button("Télécharger PDF", response.choices[0].text, file_name=f"lettre_{etablissement['etablissement']}.pdf")

if __name__ == "__main__":
    main()