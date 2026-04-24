## GitHub Copilot Chat

- Extension: 0.37.8 (prod)
- VS Code: 1.109.5 (072586267e68ece9a47aa43f8c108e0dcbf44622)
- OS: linux 6.6.87.2-microsoft-standard-WSL2 x64
- Remote Name: wsl
- Extension Kind: Workspace
- GitHub Account: palybi002

## Network

User Settings:
```json
  "http.proxy": "http://127.0.0.1:7890",
  "http.proxyStrictSSL": false,
  "http.systemCertificatesNode": true,
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.205.243.168 (34 ms)
- DNS ipv6 Lookup: Error (12 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: http://127.0.0.1:7890 (1 ms)
- Proxy Connection: 200 Connection established (10 ms)
- Electron fetch: Unavailable
- Node.js https: HTTP 200 (411 ms)
- Node.js fetch (configured): HTTP 200 (387 ms)

Connecting to https://api.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.113.21 (6 ms)
- DNS ipv6 Lookup: Error (11 ms): getaddrinfo ENOTFOUND api.githubcopilot.com
- Proxy URL: http://127.0.0.1:7890 (0 ms)
- Proxy Connection: 200 Connection established (5 ms)
- Electron fetch: Unavailable
- Node.js https: HTTP 200 (1014 ms)
- Node.js fetch (configured): 