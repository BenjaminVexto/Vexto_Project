# src/vexto/scoring/log_utils.py
from __future__ import annotations
import logging, re

# Mønstre: Google keys, generiske key-parametre, auth-headere, creds i URL
_PATTERNS = [
    # key=AIza... hvor som helst (variabel længde 20–100)
    (re.compile(r'(?i)\bkey=AIza[0-9A-Za-z\-_]{20,100}'), 'key=****'),

    # URL-encodet key (key%3DAIza...)
    (re.compile(r'(?i)key%3DAIza[0-9A-Za-z\-_]{20,100}'), 'key%3D****'),

    # Rå AIza-nøgle uden "key=" (sidste sikkerhedsnet)
    (re.compile(r'AIza[0-9A-Za-z\-_]{20,100}'), 'AIza****'),

    # Generiske query-parametre for nøgler/tokens
    (re.compile(r'(?i)([?&])(api[_-]?key|x-api-key|access[_-]?key|token|access_token|signature)=([^&\s]+)'),
     lambda m: f"{m.group(1)}{m.group(2)}=****"),

    # Authorization headers
    (re.compile(r'(?i)Authorization:\s*Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*'), 'Authorization: Bearer ****'),
    (re.compile(r'(?i)Authorization:\s*Basic\s+[A-Za-z0-9\+\/=]+'), 'Authorization: Basic ****'),

    # Credentials i URL (http://user:pass@host)
    (re.compile(r'(?i)://([^:@/\s]+):([^@/\s]+)@'), r'://\1:****@'),
]

def sanitize_message(msg: str) -> str:
    if not isinstance(msg, str):
        try:
            msg = str(msg)
        except Exception:
            return msg
    for pat, repl in _PATTERNS:
        msg = pat.sub(repl, msg)
    return msg

class MaskSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Sanitér den endelige tekst – og nulstil args for at undgå reformattering
            record.msg = sanitize_message(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True

def install_log_masking(logger: logging.Logger | None = None) -> None:
    logger = logger or logging.getLogger()  # root
    f = MaskSecretsFilter()
    logger.addFilter(f)                    # <- NYT: logger-level filter
    for h in logger.handlers:
        h.addFilter(f)                     # stadig på eksisterende handlers