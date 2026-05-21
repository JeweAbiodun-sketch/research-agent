# n8n Integration

This project returns a structured JSON payload from `/research`, plus a generated PDF report on disk.

## Recommended n8n Flow

1. `Telegram Trigger`
2. `Extract Input`
3. `Code in JavaScript`
4. `Valid Input`
5. `Call LangGraph Agent`
6. `Code in JavaScript1`
7. `Package as Binary`
8. `Telegram — Deliver Report`

## Keep The LangGraph Response Intact

If your post-LangGraph `Code in JavaScript` node rebuilds the object, it can accidentally drop fields like:

- `report_url`
- `report_file_path`
- `report_file_name`
- `logo_source`
- `logo_found`
- `token_usage`
- `estimated_cost_usd`

Use a pass-through script so the response stays intact:

```javascript
return items.map((item) => ({
  json: {
    ...item.json,
  },
  binary: item.binary ?? {},
}));
```

If you only want the first item:

```javascript
const item = items[0];

return [
  {
    json: {
      ...item.json,
    },
    binary: item.binary ?? {},
  },
];
```

## Package As Binary

Use `report_file_path` as the source for the PDF attachment if your workflow needs to reattach the generated report.

Example logic for a Code node before `Package as Binary`:

```javascript
const reportPath = $json.report_file_path;

if (!reportPath) {
  throw new Error('Missing report_file_path from LangGraph response');
}

return [
  {
    json: {
      ...$json,
      report_file_path: reportPath,
    },
  },
];
```

If your workflow already has the PDF as binary data, keep the binary property untouched and only pass through JSON.

## What The Server Now Returns

The `/research` endpoint returns these useful fields:

- `report_url`
- `report_file_path`
- `report_file_name`
- `logo_source`
- `logo_found`
- `token_usage`
- `estimated_cost_usd`
- `cost_breakdown`

That means your n8n workflow can:

- send the user a clickable report link
- attach the PDF directly
- show whether a real logo was found
- surface token and cost information in the final message

