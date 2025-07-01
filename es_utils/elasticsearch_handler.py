def _make_es_query(branche: str, postnumre: List[int], include_active_only: bool = True) -> Dict[str, Any]:
    """
    Bygger ElasticSearch forespørgslen for en given (renset) branchekode og liste af postnumre.
    Inkluderer filtrering for kun aktive virksomheder, hvis angivet.
    
    BEMÆRK: Denne funktion bygger KUN den del af query'en, der går i JSON-body.
    'size' og 'scroll' parametre håndteres separat som URL-parametre for initial request.

    Args:
        branche (str): Den rensede branchekode (f.eks. "433200").
        postnumre (list[int]): En liste af postnumre at søge efter.
        include_active_only (bool): Om der kun skal søges efter aktive virksomheder.

    Returns:
        dict: ElasticSearch forespørgsels-body i JSON-format.
    """
    # Konverter postnumre til strenge, da ElasticSearch ofte gemmer dem som strenge,
    # og 'terms' forespørgsler kræver typematch.
    postnumre_str = [str(p) for p in postnumre]

    # 'filter' klausulen i ElasticSearch er god til caching og bruges til Bool-queries,
    # hvor alle betingelser skal være sande.
    bool_query_filters = [
        {"term": {"Vrvirksomhed.virksomhedMetadata.nyesteHovedbranche.branchekode": branche}},
        {"terms": {"Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.postnummer": postnumre_str}}
    ]
    
    # Tilføj et filter for statusKode = "NORMAL" for at sikre, at kun aktive virksomheder inkluderes.
    # Dette er mere robust end at udelukke specifikke tekstbeskrivelser.
    if include_active_only:
        bool_query_filters.append({
            "term": {
                "Vrvirksomhed.virksomhedMetadata.nyesteVirksomhedsstatus.statusKode": "NORMAL"
            }
        })

    es_query_body = {
        # '_source' definerer, hvilke felter der skal returneres i svaret.
        # Kun de nødvendige felter bør inkluderes for at minimere datamængden.
        "_source": [
            "Vrvirksomhed.cvrNummer",
            "Vrvirksomhed.virksomhedMetadata.nyesteNavn.navn",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.postnummer",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.vejnavn",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.husnummerFra",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.postdistrikt",
            "Vrvirksomhed.virksomhedMetadata.nyesteHovedbranche.branchekode",
            "Vrvirksomhed.virksomhedMetadata.nyesteHovedbranche.branchetekst",
            "Vrvirksomhed.virksomhedMetadata.nyesteVirksomhedsstatus.statusTekst" # Inkluder status for lettere debug
        ],
        "query": {
            "bool": {
                "filter": bool_query_filters # Disse skal matche
            }
        },
        "sort": [{"_doc": "asc"}] # Tilføj sortering for at sikre ensartet rækkefølge, vigtigt for search_after og duplikathåndtering.
    }
        
    return es_query_body