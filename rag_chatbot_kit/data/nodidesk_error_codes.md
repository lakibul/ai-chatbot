# NodiDesk — API Error Codes

| Code | HTTP status | Meaning | What to do |
|---|---|---|---|
| ND-1001 | 401 | Invalid or missing API key | Check the X-Nodi-Key header |
| ND-1042 | 429 | Rate limit exceeded | Wait for the number of seconds in the Retry-After header, then retry |
| ND-2003 | 404 | Ticket not found | Check the ticket ID; the ticket may have been deleted |
| ND-2010 | 413 | Attachment too large | Files must be 25 MB or smaller |
| ND-3005 | 401 | SSO assertion failed | Ask your identity provider admin to check the SAML certificate |
| ND-4000 | 500 | Internal server error | Contact support and include the request ID from the X-Request-Id header |
