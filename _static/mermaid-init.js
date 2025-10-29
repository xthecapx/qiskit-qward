document.addEventListener('DOMContentLoaded', function() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            themeVariables: {
                primaryColor: '#ff6b6b',
                primaryTextColor: '#333',
                primaryBorderColor: '#ff6b6b',
                lineColor: '#333',
                secondaryColor: '#4ecdc4',
                tertiaryColor: '#ffe66d'
            },
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            },
            classDiagram: {
                useMaxWidth: true
            }
        });
        
        // Force re-render of mermaid diagrams
        setTimeout(function() {
            mermaid.run();
        }, 100);
    }
}); 