import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

with open('evaluation/dataset.json', encoding='utf-8') as f:
    dataset = json.load(f)

mock_responses = {
    'risque': 'Les machines a risque eleve ont une probabilite de panne superieure a 50 pourcent.',
    'combien': 'Le systeme surveille 20 machines au total.',
    'alerte': 'Il y a 3 alertes critiques actives necessitant une intervention immediate.',
    'niveau de risque': 'Le niveau de risque est calcule par le modele ML selon les metriques collectees.',
    'maintenance': 'La maintenance predictive anticipe les pannes grace au Machine Learning.',
    'probabilite': 'Une probabilite de 80 pourcent indique un risque tres eleve dans les 30 prochains jours.',
    'critical': 'Une alerte CRITICAL necessite une intervention immediate.',
    'high': 'HIGH signifie risque eleve entre 50 et 70 pourcent, CRITICAL superieur a 70 pourcent.',
    'collecte': 'L agent Python collecte les metriques CPU RAM et disque toutes les heures.',
    'smart': 'Les donnees SMART surveillent la sante des disques durs.',
    'modele': 'Le modele LSTM analyse les metriques pour predire les pannes.',
    'cpu': 'Si le CPU depasse 90 pourcent, identifiez les processus consommateurs.',
    'prediction': 'Les predictions sont mises a jour chaque nuit a 2h00 du matin.',
    'reduire': 'Effectuez une maintenance preventive et remplacez les composants vieillissants.',
    'disque': 'Un disque a 95 pourcent est critique, liberez de l espace immediatement.',
    'email': 'Les alertes email sont envoyees via Nodemailer pour les niveaux HIGH et CRITICAL.',
    'frequence': 'Les metriques sont collectees toutes les heures par l agent Python.',
    'tableau': 'Le tableau de bord est accessible sur http://localhost:5173 apres connexion.',
    'anomalie': 'Une anomalie est un comportement inhabituel detecte par Isolation Forest.',
    'graphique': 'Les graphiques montrent l evolution des metriques CPU RAM et disque dans le temps.',
}

def get_mock(q):
    q_lower = q.lower()
    for kw, resp in mock_responses.items():
        if kw in q_lower:
            return resp
    return 'Je suis votre assistant de maintenance predictive.'

sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1

results = []
for item in dataset:
    q = item['question']
    ref = item['reference']
    hyp = get_mock(q)
    ref_tok = nltk.word_tokenize(ref.lower())
    hyp_tok = nltk.word_tokenize(hyp.lower())
    bleu = sentence_bleu([ref_tok], hyp_tok, smoothing_function=smoother)
    rouge = sc.score(ref, hyp)
    results.append({
        'id': item['id'],
        'q': q[:42],
        'bleu': bleu,
        'r1': rouge['rouge1'].fmeasure,
        'r2': rouge['rouge2'].fmeasure,
        'rL': rouge['rougeL'].fmeasure
    })

print("=" * 78)
print("  CHATBOT EVALUATION — MOCK MODE — BLEU & ROUGE SCORES")
print("=" * 78)
print("{:<4} {:<42} {:>6} {:>6} {:>6} {:>6}".format("#", "Question", "BLEU", "R-1", "R-2", "R-L"))
print("-" * 78)
for r in results:
    print("{:<4} {:<42} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f}".format(
        r['id'], r['q'], r['bleu'], r['r1'], r['r2'], r['rL']))
print("-" * 78)
n = len(results)
avg_bleu = sum(r['bleu'] for r in results) / n
avg_r1   = sum(r['r1']   for r in results) / n
avg_r2   = sum(r['r2']   for r in results) / n
avg_rL   = sum(r['rL']   for r in results) / n
print("{:<4} {:<42} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f}".format(
    "AVG", "", avg_bleu, avg_r1, avg_r2, avg_rL))
print("=" * 78)
print("")
print("AVERAGE SCORES SUMMARY:")
print("  BLEU    : {:.4f}".format(avg_bleu))
print("  ROUGE-1 : {:.4f}".format(avg_r1))
print("  ROUGE-2 : {:.4f}".format(avg_r2))
print("  ROUGE-L : {:.4f}".format(avg_rL))
print("=" * 78)
