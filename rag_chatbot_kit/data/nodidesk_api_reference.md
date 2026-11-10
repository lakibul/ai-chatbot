# NodiDesk — REST API Reference (v2)

Base URL: https://api.nodidesk.example/v2
The API is available on the Growth and Enterprise plans only.

## Authentication
Send your API key in the X-Nodi-Key header with every request. Create keys under Settings > API Keys.
The old X-Api-Key header was removed in release 2.6.

## Endpoints
| Method | Path | Description |
|---|---|---|
| POST | /tickets | Create a ticket |
| GET | /tickets/{id} | Get one ticket |
| PATCH | /tickets/{id} | Update status, priority or assignee |
| GET | /tickets?status=open | List tickets, filtered by status |

## Pagination
List endpoints use cursor pagination. Pass the cursor value from the previous response as ?cursor=. The maximum page size is 100 (?limit=100).

## Rate limits
Growth: 60 requests per minute. Enterprise: 600 requests per minute. Exceeding the limit returns error ND-1042.

## Webhooks
NodiDesk can send events (ticket.created, ticket.updated, ticket.closed) to your URL.
Each webhook includes an X-Nodi-Signature header: an HMAC-SHA256 of the request body, signed with your webhook secret. Always verify it before trusting the event.
