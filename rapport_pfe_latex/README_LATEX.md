# Comment compiler ce rapport sur Overleaf

## Structure des fichiers

```
rapport_pfe_latex/
├── main.tex                    ← Fichier principal (compiler celui-ci)
├── references.bib              ← Bibliographie BibLaTeX
├── images/                     ← ⚠️ METTRE VOS IMAGES ICI
│   ├── class_diagram.png               ← Diagramme de classes (UML)
│   ├── use_case_diagram.png            ← Diagramme de cas d'utilisation
│   ├── sequence_dashboard.png          ← Séquence : affichage dashboard
│   ├── sequence_prediction_alert.png   ← Séquence : génération prédictions
│   ├── sequence_data_collection.png    ← Séquence : collecte agent
│   ├── screen_dashboard.png            ← Capture : tableau de bord principal
│   ├── screen_machine_details.png      ← Capture : détail machine + prédictions
│   ├── screen_alerts.png               ← Capture : gestion des alertes
│   ├── screen_chatbot.png              ← Capture : assistant chatbot
│   ├── screen_model_performance.png    ← Capture : évaluation modèles ML
│   ├── screen_user_management.png      ← Capture : gestion utilisateurs (admin)
│   ├── screen_landing.png              ← Capture : page d'accueil publique
│   └── screen_login.png                ← Capture : page de connexion
│   ├── screen_signup.png               ← Capture : page d'inscription
│   └── screen_forgot_password.png      ← Capture : réinitialisation mot de passe
├── pages/
│   ├── titlepage.tex
│   ├── acknowledgements.tex
│   ├── dedication.tex
│   ├── webography.tex
│   └── acronyms.tex
└── chapters/
    ├── introduction.tex
    ├── chapter1_overview.tex
    ├── chapter2_literature.tex
    ├── chapter3_design.tex
    ├── chapter4_implementation.tex
    └── conclusion.tex
```

## Instructions Overleaf

1. Créer un nouveau projet sur https://overleaf.com
2. Uploader tous les fichiers en conservant la structure de dossiers
3. **Créer un dossier `images/` et uploader les 5 PNG dedans**
4. Dans les paramètres du projet : Compilateur = **pdfLaTeX**, BibTeX = **Biber**
5. Compiler `main.tex`

## Personnalisation

- Remplacer `[Prénom NOM]` par ton nom dans `titlepage.tex`
- Remplacer `[Prénom NOM Encadrant]` par le nom de ton encadrant
- Ajouter ton logo université : décommenter la ligne `\includegraphics` dans `titlepage.tex`
