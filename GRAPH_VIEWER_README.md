# Knowledge Graph Viewer

A simplified, working knowledge graph visualization for the LennyHub RAG system.

## What's New

This is a simpler, more reliable version that:
- ✅ Uses D3.js force-directed graph (lighter & faster)
- ✅ Dark theme that's easy on the eyes
- ✅ Better error handling and loading states
- ✅ Works with large graphs (5000+ nodes)
- ✅ Interactive controls (drag, zoom, search, click)
- ✅ Built-in Python server for easy viewing

## Quick Start

### Option 1: Using Python Server (Recommended)

```bash
# Start the server
python serve_graph.py

# The graph will open automatically in your browser
# If not, visit: http://localhost:8000/graph_viewer_simple.html
```

### Option 2: Using Any HTTP Server

```bash
# Using Python's built-in server
python -m http.server 8000

# Then open: http://localhost:8000/graph_viewer_simple.html
```

**Important**: Don't open the HTML file directly (file:// protocol) - it needs to be served via HTTP to load the JSON data.

## Features

### Interactive Controls
- **Drag nodes**: Click and drag any node to reposition it
- **Zoom**: Scroll wheel to zoom in/out
- **Pan**: Drag the background to move around
- **Search**: Type in the search box to find and highlight entities
- **Click node**: View detailed information in the sidebar
- **Double-click node**: Focus and zoom to that node

### Sidebar
- **Stats**: Total nodes and links
- **Search**: Find entities by name or description
- **Reset View**: Return to default zoom and position
- **Restart Simulation**: Re-run the physics simulation
- **Node Details**: Click any node to see its full information

### Color Coding
- 🔴 **Red**: Person
- 🔵 **Blue**: Organization/Platform
- 🟣 **Purple**: Concept
- 🟠 **Orange**: Method
- 🟢 **Green**: Product
- ⚫ **Gray**: Other types

## Performance Notes

- The graph automatically limits edges to 5,000 for performance
- Labels are shown for the first 100 nodes only
- Hover over any node to see its name
- The simulation will stabilize after a few seconds

## Files

- `graph_viewer_simple.html` - The main visualization file
- `graph_data.json` - Your knowledge graph data
- `serve_graph.py` - Simple Python server to view the graph

## Troubleshooting

### Graph won't load
- Make sure `graph_data.json` exists in the same directory
- Ensure you're using HTTP (not file://) to serve the page
- Check browser console (F12) for error messages

### Performance issues
- The graph limits edges automatically
- Try restarting the simulation with the button
- Close other browser tabs
- Disable the physics simulation by commenting out the force simulation restart

### Can't see all nodes
- Use the search function to find specific entities
- Zoom out using the Zoom - button or scroll wheel
- Reset the view with the Reset View button

## Technical Details

- **Visualization**: D3.js v7 force-directed graph
- **Data format**: JSON with nodes and edges arrays
- **Browser compatibility**: Modern browsers (Chrome, Firefox, Safari, Edge)
- **File size**: Optimized for graphs up to 10,000 nodes

## Next Steps

Want to customize the visualization? Edit `graph_viewer_simple.html`:
- Line 24-46: Adjust colors and styling
- Line 417-420: Modify force simulation parameters
- Line 442-444: Change node appearance
- Line 486-494: Customize search behavior

Enjoy exploring your knowledge graph! 🎉
