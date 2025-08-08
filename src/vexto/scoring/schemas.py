# src/vexto/scoring/schemas.py

from typing import TypedDict, NotRequired, List, Optional, Dict

__all__ = [
    "BasicSEO", "TechnicalSEO", "PerformanceMetrics", "ContentMetrics",
    "AuthorityMetrics", "SocialAndReputationMetrics", "ConversionMetrics",
    "PrivacyMetrics", "SecurityMetrics", "UrlAnalysisData"
]

class BasicSEO(TypedDict):
    """KPI'er relateret til on-page HTML-tags."""
    h1: Optional[str]
    h1_count: Optional[int]
    h1_texts: Optional[List[str]]
    meta_description: Optional[str]
    meta_description_length: Optional[int]
    title_text: Optional[str]
    title_length: Optional[int]
    word_count: Optional[int]
    image_count: Optional[int]
    image_alt_count: Optional[int]
    image_alt_pct: Optional[int]
    avg_image_size_kb: Optional[int]
    canonical_url: Optional[str]
    canonical_error: Optional[str]
    schema_markup_found: bool

class TechnicalSEO(TypedDict):
    """KPI'er relateret til server-setup og teknisk sundhed."""
    status_code: Optional[int]
    is_https: bool
    robots_txt_found: bool
    sitemap_xml_found: bool
    sitemap_is_fresh: bool # <-- TILFØJET
    sitemap_locations: List[str]
    response_time_ms: Optional[float]
    total_pages_crawled: Optional[int]
    total_links_found: Optional[int]
    broken_links_count: Optional[int]
    broken_links_pct: Optional[int]
    broken_links_list: Optional[Dict[str, int]]
    
class PerformanceMetrics(TypedDict):
    """KPI'er relateret til Core Web Vitals og sidehastighed."""
    lcp_ms: Optional[float]
    cls: Optional[float]
    inp_ms: Optional[int]
    viewport_score: Optional[int] # <-- TILFØJET
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
    gmb_review_count: Optional[int]
    gmb_average_rating: Optional[float]
    gmb_profile_complete: Optional[bool]
    social_media_links: List[str]
    
class ConversionMetrics(TypedDict, total=False):
    """KPI'er relateret til sporing og konverteringspotentiale."""
    has_ga4: Optional[bool]
    has_meta_pixel: Optional[bool]
    emails_found: List[str]
    phone_numbers_found: List[str]
    form_field_counts: List[int]
    trust_signals_found: List[str]
    
class PrivacyMetrics(TypedDict, total=False):
    """GDPR / Privacy-relaterede checks."""
    cookie_banner_detected: Optional[bool]
    detection_method: Optional[str]
    personal_data_redacted: NotRequired[bool]

class SecurityMetrics(TypedDict, total=False):
    """KPI'er relateret til HTTP Security Headers."""
    hsts_enabled: bool
    csp_enabled: bool
    x_content_type_options_enabled: bool
    x_frame_options_enabled: bool

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