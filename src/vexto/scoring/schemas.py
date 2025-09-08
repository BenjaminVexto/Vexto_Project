# src/vexto/scoring/schemas.py

from typing import TypedDict, NotRequired, List, Optional, Dict




__all__ = [
    "BasicSEO", "TechnicalSEO", "PerformanceMetrics", "ContentMetrics",
    "AuthorityMetrics", "SocialAndReputationMetrics", "ConversionMetrics",
    "PrivacyMetrics", "SecurityMetrics", "UrlAnalysisData"
]


class BasicSEO(TypedDict, total=False):
    h1: Optional[str]
    h1_count: int
    h1_texts: List[str]
    meta_description: Optional[str]
    meta_description_length: int
    title_text: Optional[str]
    title_length: int
    word_count: int
    canonical_url: Optional[str]
    canonical_error: Optional[str]
    schema_markup_found: bool
    schema_types: List[str]
    canonical_source: str

    # NYT: JSON-LD/microdata-telemetri
    schema_jsonld_valid: NotRequired[bool]     # mindst ét JSON-LD parsed OK
    jsonld_repaired: NotRequired[bool]         # mindst ét script blev "repareret" før parse
    schema_microdata_found: NotRequired[bool]  # itemscope fundet

class TechnicalSEO(TypedDict):
    """KPI'er relateret til server-setup og teknisk sundhed."""
    status_code: Optional[int]
    is_https: bool
    robots_txt_found: bool
    sitemap_xml_found: bool
    sitemap_is_fresh: bool
    sitemap_locations: List[str]
    response_time_ms: Optional[float]
    total_pages_crawled: Optional[int]
    total_links_found: Optional[int]
    broken_links_count: Optional[int]
    broken_links_pct: Optional[int]
    broken_links_list: Optional[Dict[str, int]]
    # Debug / forkastede links (sample af <20)
    discarded_links: NotRequired[List[Dict[str, str]]]
    # --- NYT: Render-telemetri & soft-404 ---
    render_status: Optional[str]  # "content" | "empty"
    rendered_content_length: Optional[int]  # antal tegn i renderet HTML
    soft_404_suspected: Optional[bool]  # True hvis 404 + har indhold
    
class PerformanceMetrics(TypedDict):
    """KPI'er relateret til Core Web Vitals og sidehastighed."""
    lcp_ms: Optional[float]                        # effektiv (cappet) LCP
    lcp_ms_raw: NotRequired[float]                 # NY: rå LCP før cap
    performance_source: NotRequired[str]           # NY: "field" | "lab"
    performance_strategy: NotRequired[str]         # NY: "mobile" | "desktop" | "unknown"
    cls: Optional[float]
    inp_ms: Optional[int]
    viewport_score: Optional[int]                  # mobil-venlig (0/1) fra audit
    performance_score: Optional[int]
    psi_status: str
    total_js_size_kb: Optional[int]
    js_file_count: Optional[int]

class ContentMetrics(TypedDict, total=False):
    """KPI'er relateret til indholdets kvalitet og relevans."""
    latest_post_date: Optional[str]
    keywords_in_content: dict
    internal_link_score: Optional[int]

class AuthorityMetrics(TypedDict):
    """KPI'er relateret til domæneautoritet og omdømme (eksterne)."""
    domain_authority: Optional[float]
    page_authority: Optional[float]
    global_rank: Optional[int]
    authority_status: str

class SocialAndReputationMetrics(TypedDict, total=False):
    """KPI'er relateret til omdømme på sociale medier og Google."""
    # Google Business Profile (GMB)
    gmb_review_count: Optional[int]
    gmb_average_rating: Optional[float]
    gmb_profile_complete: Optional[bool]
    # NYT: status for søgning/tilgængelighed
    #  - "ok"      → profil fundet (data udfyldt)
    #  - "unknown" → intet sikkert match / manglende API-key / ZERO_RESULTS
    gmb_status: Optional[str]

    # (Findes i koden/bruges i YAML – eksplicit typer for bedre statisk checking)
    gmb_has_website: Optional[bool]
    gmb_has_hours: Optional[bool]
    gmb_photo_count: Optional[int]
    gmb_business_name: Optional[str]
    gmb_address: Optional[str]
    gmb_place_id: Optional[str]

    # Social
    social_media_links: List[str]
    
class ConversionMetrics(TypedDict, total=False):
    """KPI'er relateret til sporing og konverteringspotentiale."""
    # Eksisterende
    has_ga4: Optional[bool]
    has_meta_pixel: Optional[bool]
    has_gtm: Optional[bool]

    # GA4 / Tag-telemetri
    ga4_measurement_id: Optional[str]              # f.eks. "G-ABC12345"
    ga4_measurement_id_source: Optional[str]       # "html" | "runtime_gtag" | "runtime_send_to" | "inferred"
    google_tag_id: Optional[str]                   # f.eks. "GT-ABC12345"
    google_ads_send_to: List[str]

    # Kontaktdata (høj sikkerhed)
    emails_found: List[str]
    phone_numbers_found: List[str]

    # Kontaktdata (lav sikkerhed) — NYT
    emails_low_confidence: NotRequired[List[str]]
    phone_numbers_low_confidence: NotRequired[List[str]]

    # Formularer og trust
    form_field_counts: List[int]
    trust_signals_found: List[str]
    
class PrivacyMetrics(TypedDict, total=False):
    """GDPR / Privacy-relaterede checks."""
    cookie_banner_detected: Optional[bool]
    detection_method: Optional[str]
    # NYT: graderet sikkerhed for detektion
    cookie_banner_confidence: NotRequired[float]  # 0.0–1.0
    personal_data_redacted: NotRequired[bool]

class SecurityMetrics(TypedDict, total=False):
    """KPI'er relateret til HTTP Security Headers."""
    hsts_enabled: bool
    hsts_include_subdomains: NotRequired[bool]
    hsts_preload: NotRequired[bool]
    csp_enabled: bool
    x_content_type_options_enabled: bool
    # Tæl også 'frame-ancestors' i CSP som frame-beskyttelse
    x_frame_options_enabled: bool
    # Notér om målingen blev foretaget på HEAD eller GET
    measured_on: NotRequired[str]  # 'HEAD' | 'GET'

class UrlAnalysisData(TypedDict):
    """Den samlede datakontainer for en enkelt URL."""
    url: str
    fetch_error: NotRequired[str]
    fetch_method: NotRequired[str]
    basic_seo: BasicSEO
    technical_seo: TechnicalSEO
    performance: PerformanceMetrics
    authority: AuthorityMetrics
    security: NotRequired[SecurityMetrics]
    content: NotRequired[ContentMetrics]
    social_and_reputation: NotRequired[SocialAndReputationMetrics]
    conversion: NotRequired[ConversionMetrics]
    privacy: NotRequired[PrivacyMetrics]