# Free Stuff Page

Static HTML page for giving away items. No build step required.

## Adding/Editing Items

Edit the `items` array in `index.html` (around line 230). Each item:

```json
{
  "id": "unique-id",
  "title": "Item Name",
  "description": "Description text.",
  "images": ["imgs/photo1.jpg", "imgs/photo2.jpg"],
  "available": true
}
```

- Set `available: false` to mark as taken
- Images go in `imgs/` directory
- Multiple images get automatic carousel navigation
- the data json is inline in the HTML.
