# tests/test_privacy_fetchers.py
import pytest
from bs4 import BeautifulSoup
from vexto.scoring.privacy_fetchers import detect_cookie_banner

HTML_WITH_BANNER_CLASS = """
<html><body>
  <div class="cookie-consent-banner">
    <p>We use cookies. <button>Accept</button></p>
  </div>
</body></html>
"""

HTML_WITH_BANNER_ID = """
<html><body>
  <div id="user-consent-dialog">
    <p>Please accept.</p>
  </div>
</body></html>
"""

HTML_WITH_TEXT = """
<html><body>
  <p>To continue, please accept our use of cookies to comply with GDPR regulations.</p>
</body></html>
"""

HTML_CLEAN = """
<html><body>
  <h1>Welcome!</h1>
  <p>No banners here. Read our privacy policy in the footer.</p>
</body></html>
"""

def test_detect_banner_by_selector_class():
    soup = BeautifulSoup(HTML_WITH_BANNER_CLASS, 'lxml')
    result = detect_cookie_banner(soup)
    assert result['cookie_banner_detected'] is True
    assert result['detection_method'] == 'selector:cookie'

def test_detect_banner_by_selector_id():
    soup = BeautifulSoup(HTML_WITH_BANNER_ID, 'lxml')
    result = detect_cookie_banner(soup)
    assert result['cookie_banner_detected'] is True
    assert result['detection_method'] == 'selector:consent'

def test_detect_banner_by_text():
    soup = BeautifulSoup(HTML_WITH_TEXT, 'lxml')
    result = detect_cookie_banner(soup)
    assert result['cookie_banner_detected'] is True
    assert result['detection_method'] == 'text:cookie'

def test_no_banner_detected():
    soup = BeautifulSoup(HTML_CLEAN, 'lxml')
    result = detect_cookie_banner(soup)
    assert result['cookie_banner_detected'] is False
    assert result['detection_method'] == 'none'

def test_none_soup():
    result = detect_cookie_banner(None)
    assert result['cookie_banner_detected'] is None
    assert result['detection_method'] == 'no_html'