document.addEventListener("DOMContentLoaded", function() {
  
  // --- 1. Copy Code Button ---
  const codeBlocks = document.querySelectorAll('div.highlighter-rouge');
  
  codeBlocks.forEach(block => {
    // Create the button container
    const header = document.createElement('div');
    header.className = 'code-header';
    
    const button = document.createElement('button');
    button.className = 'copy-btn';
    button.textContent = 'Copy';
    
    header.appendChild(button);
    
    // Insert before the actual code block (pre)
    block.insertBefore(header, block.firstChild);
    
    button.addEventListener('click', () => {
      // Find the code text
      const code = block.querySelector('code').innerText;
      
      navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        button.classList.add('copied');
        
        setTimeout(() => {
          button.textContent = 'Copy';
          button.classList.remove('copied');
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy: ', err);
      });
    });
  });

  // --- 2. Heading Anchors ---
  const headings = document.querySelectorAll('.post-content h2, .post-content h3, .post-content h4');
  
  headings.forEach(heading => {
    if (heading.id) {
      const anchor = document.createElement('a');
      anchor.className = 'anchor-link';
      anchor.href = '#' + heading.id;
      anchor.textContent = '#';
      anchor.title = "Link to this section";
      
      heading.appendChild(anchor);
    }
  });
});
