# ---- Partie Backend ----
class StudentAnalyzer:
    def __init__(self):
        self.riasec_mapping = {
            'R': 'Réaliste', 'I': 'Investigateur', 'A': 'Artistique',
            'S': 'Social', 'E': 'Entreprenant', 'C': 'Conventionnel'
        }

    def parse_cv(self, pdf_file):
        # ... (méthode existante inchangée)

    def _process_grades(self, pdf_file):  # CORRECTION : Bonne indentation dans la classe
        """Traite les relevés de notes PDF"""
        import re  # Module standard, pas besoin de l'installer
        import camelot
        
        if pdf_file is None:
            return {}

        try:
            # Extraction texte
            text = self.parse_cv(pdf_file)
            
            # Extraction des tableaux
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(pdf_file.getvalue())
                tables = camelot.read_pdf(tmp.name, flavor='stream')
                df = pd.concat([t.df for t in tables])
            
            # Logique de traitement...
            return {
                'moyenne': ...,
                'matieres_fortes': [...],
                'matieres_faibles': [...]
            }
        except Exception as e:
            st.error(f"Erreur : {str(e)}")
            return {}

class EstablishmentMatcher:
    def _load_etablissements_data(self):
        """Charge les données des établissements"""
        response = requests.get("https://data.enseignementsup-recherche.gouv.fr/pages/home/")
        # CORRECTION : Utilisation des bonnes clés
        return pd.DataFrame([{
            'etablissement': item.get('fields', {}).get('etablissement_lib'),
            'ville': item.get('fields', {}).get('commune_lib'),
            'formation': item.get('fields', {}).get('form_lib_voe_acc')
        } for item in response.json().get('records', [])])