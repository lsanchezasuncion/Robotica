import os
import sys
import requests
import schedule
import time
import argparse
import warnings

from dotenv import load_dotenv
from transformers import pipeline, logging as transformers_logging
from bs4 import BeautifulSoup

# 1. SILENCIAR WARNINGS Y LOGS DE TRANSFORMERS
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

# 2. CARGAR VARIABLES DE ENTORNO (NEWSAPI_KEY)
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# 3. CREAR PIPELINE DE RESUMEN (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 4. FUNCION AUXILIAR PARA RESUMIR SIN VER WARNINGS "max_length"
def dynamic_summarize(text, min_len=100, max_len=450):
    """
    Ajusta max_length dinamicamente según la longitud del texto
    para evitar el warning "Your max_length is set to X, but your input_length is only Y..."
    """
    word_count = len(text.split())
    # Cap max_len al 80% del numero de palabras (heuristica)
    estimated_max_len = min(max_len, int(word_count * 0.8))

    # Asegurar que no sea menor que min_len
    if estimated_max_len < min_len:
        estimated_max_len = min_len + 5

    result = summarizer(
        text,
        max_length=estimated_max_len,
        min_length=min_len,
        do_sample=False
    )
    return result[0]['summary_text']


# ------------------------------------------------------------------------------
# 5. FUNCIONES PARA OBTENER NOTICIAS
# ------------------------------------------------------------------------------

def fetch_from_newsapi(topic, language="es", page_size=5):
    """
    Llama a la API de NewsAPI sin restringir dominios (fuente 'newsapi').
    """
    if not NEWSAPI_KEY:
        print("No se encontró NEWSAPI_KEY en .env. No se pueden obtener noticias de NewsAPI.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt"
    }
    headers = {
        "Authorization": f"Bearer {NEWSAPI_KEY}"
    }

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        print("Error al obtener noticias de NewsAPI:", resp.text)
        return []

    data = resp.json()
    articles = data.get("articles", [])
    results = []
    for art in articles:
        results.append({
            "title": art.get("title", ""),
            "description": art.get("description", ""),
            "url": art.get("url", "")
        })
    return results


def fetch_from_lista(topic, language="es", page_size=5, domain_string=None):
    """
    Llama a la API de NewsAPI filtrando SOLO los dominios indicados en domain_string.
    domain_string: "bbc.co.uk, washingtonpost.com, lemonde.fr, elpais.com, elmundo.es, europapress.es"
    """
    if not NEWSAPI_KEY:
        print("No se encontró NEWSAPI_KEY en .env. No se pueden obtener noticias de NewsAPI.")
        return []

    if not domain_string:
        # Por si el usuario no pasa nada, set de dominios internacionales por defecto
        domain_string = "bbc.co.uk,washingtonpost.com,lemonde.fr,elpais.com,elmundo.es,europapress.es"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "domains": domain_string
    }
    headers = {
        "Authorization": f"Bearer {NEWSAPI_KEY}"
    }

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        print("Error al obtener noticias de la 'lista' de periódicos:", resp.text)
        return []

    data = resp.json()
    articles = data.get("articles", [])
    results = []
    for art in articles:
        results.append({
            "title": art.get("title", ""),
            "description": art.get("description", ""),
            "url": art.get("url", "")
        })
    return results


def fetch_news(topic, language="es", page_size=5, source="newsapi", domain_string=None):
    """
    Selecciona la fuente: 'newsapi' o 'lista'
    Si es 'lista', usamos fetch_from_lista() con 'domains'.
    """
    if source == "newsapi":
        return fetch_from_newsapi(topic, language, page_size)
    elif source == "lista":
        return fetch_from_lista(topic, language, page_size, domain_string)
    else:
        print(f"Fuente desconocida '{source}'. Usa 'newsapi' o 'lista'.")
        return []


# ------------------------------------------------------------------------------
# 6. FILTRADO POR TIPO DE ARTICULO
# ------------------------------------------------------------------------------

def filter_by_article_type(articles, article_type):
    """
    Filtro pocho por palabras clave (title/description):
    - 'paper', 'opinion', 'divulgacion', 'agency'
    - 'all' o 'any' => sin filtrar
    """
    if article_type in ("all", "any"):
        return articles

    filtered = []
    for art in articles:
        title_lower = art["title"].lower()
        desc_lower = art["description"].lower() if art["description"] else ""

        if article_type == "divulgacion":
            if ("divulgación" in title_lower or "divulgación" in desc_lower):
                filtered.append(art)

        elif article_type == "paper":
            if ("paper" in title_lower or "study" in title_lower or "investigación" in title_lower
                or "paper" in desc_lower or "study" in desc_lower or "investigación" in desc_lower):
                filtered.append(art)

        elif article_type == "opinion":
            if ("opinión" in title_lower or "column" in title_lower or "op-ed" in title_lower
                or "opinión" in desc_lower or "column" in desc_lower or "op-ed" in desc_lower):
                filtered.append(art)

        elif article_type == "agency":
            if ("agencia" in title_lower or "reuters" in title_lower or "associated press" in title_lower
                or "ap " in title_lower or "afp" in title_lower
                or "agencia" in desc_lower or "reuters" in desc_lower
                or "ap " in desc_lower or "afp" in desc_lower):
                filtered.append(art)

    return filtered


# ------------------------------------------------------------------------------
# 7. LOGICA PRINCIPAL: RESUMIR NOTICIAS
# ------------------------------------------------------------------------------

def run_agent(args):
    topic = args.topic
    language = args.language
    page_size = args.max_results
    source = args.source
    article_type = args.article_type
    domain_string = args.lista  # Por si el usuario pasa --lista "bbc.co.uk,elpais.com,..."

    print(f"\nBuscando noticias sobre '{topic}' en idioma '{language}'...")
    print(f"Fuente: {source} | Máx resultados: {page_size} | Tipo artículo: {article_type}")

    articles = fetch_news(topic, language, page_size, source=source, domain_string=domain_string)
    filtered = filter_by_article_type(articles, article_type)

    if not filtered:
        print("No se encontraron noticias tras el filtro.")
        return

    final_summaries = []
    for i, art in enumerate(filtered, 1):
        title = art["title"]
        desc = art["description"] or ""
        url = art["url"]
        text_to_summarize = f"{title}. {desc}"

        summary_text = dynamic_summarize(text_to_summarize)

        final_summaries.append(
            f"{i}. Título: {title}\n"
            f"   URL: {url}\n"
            f"   Resumen: {summary_text}\n"
        )

    print("===== RESUMEN DE NOTICIAS =====")
    for summary in final_summaries:
        print(summary)


# ------------------------------------------------------------------------------
# 8. ARGUMENTOS DE LINEA DE COMANDOS & SCHEDULE
# ------------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Agente de noticias usando NewsAPI.\n\n"
            "Opciones:\n"
            "  - 'newsapi': búsqueda normal en todos los dominios.\n"
            "  - 'lista': restringir a ciertos dominios (BBC, WashPost, Le Monde, El País, El Mundo, etc.)\n"
            "    usando la opción --lista.\n\n"
            "Ejemplo:\n"
            "  python main.py --source lista --lista 'bbc.co.uk,elpais.com' -q 'climate change'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--time", "-t", default="09:00",
                        help="Hora de ejecución diaria (formato HH:MM). Por defecto 09:00.")
    parser.add_argument("--once", "-o", action="store_true",
                        help="Ejecuta el agente inmediatamente y luego lo programa a la hora indicada.")
    parser.add_argument("--topic", "-q", default="Cambio climático",
                        help="Tema o palabra clave a buscar. Soporta algunos operadores lógicos en NewsAPI.")
    parser.add_argument("--language", "-l", default="es",
                        help="Idioma de la búsqueda. Por defecto: 'es'.")
    parser.add_argument("--max-results", "-m", type=int, default=5,
                        help="Número máximo de resultados a obtener. Por defecto: 5.")
    parser.add_argument("--source", "-s", default="newsapi",
                        choices=["newsapi", "lista"],
                        help="Fuente de las noticias. 'newsapi' o 'lista'.")
    parser.add_argument("--article-type", "-a", default="any",
                        choices=["divulgacion", "paper", "opinion", "agency", "all", "any"],
                        help="Filtrar por tipo de artículo: 'paper', 'opinion', etc. (Por defecto: any).")
    parser.add_argument("--lista",
                        help="Cadena de dominios para 'lista'. Ej: 'bbc.co.uk,washingtonpost.com,lemonde.fr,...'")

    if len(sys.argv) == 1:
        print("No se han proporcionado argumentos. Usa:")
        print("  python main.py --help")
        sys.exit(1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.once:
        run_agent(args)

    schedule.every().day.at(args.time).do(run_agent, args)

    print(f"Iniciando el agente de noticias... Se ejecutará a diario a las {args.time}.")
    print("Presiona Ctrl+C para detener.")

    while True:
        schedule.run_pending()
        time.sleep(60)
