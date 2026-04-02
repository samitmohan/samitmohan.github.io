// --- 0. Theme Toggle (runs immediately, before DOMContentLoaded) ---
(function() {
  var saved = null;
  try {
    saved = localStorage.getItem('theme');
  } catch (e) {
    // localStorage unavailable (e.g. private browsing) - fall back to null
  }
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark-mode');
  }
})();

document.addEventListener("DOMContentLoaded", function() {

  // Sync dark-mode class to body as well (for any third-party CSS targeting body)
  if (document.documentElement.classList.contains('dark-mode')) {
    document.body.classList.add('dark-mode');
  }

  // --- 0b. Lazy-load all post images ---
  var postImages = document.querySelectorAll('.post-content img');
  postImages.forEach(function(img) {
    img.setAttribute('loading', 'lazy');
    img.setAttribute('decoding', 'async');
  });

  // Theme toggle button
  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function() {
      document.documentElement.classList.toggle('dark-mode');
      document.body.classList.toggle('dark-mode');
      var isDark = document.documentElement.classList.contains('dark-mode');
      try {
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
      } catch (e) {
        // localStorage unavailable (e.g. private browsing) - silently ignore
      }
    });
  }

  // --- 1. Copy Code Button ---
  var codeBlocks = document.querySelectorAll('div.highlighter-rouge');

  codeBlocks.forEach(function(block) {
    var header = document.createElement('div');
    header.className = 'code-header';

    var button = document.createElement('button');
    button.className = 'copy-btn';
    button.textContent = 'Copy';

    header.appendChild(button);
    block.insertBefore(header, block.firstChild);

    button.addEventListener('click', function() {
      var code = block.querySelector('code').innerText;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(code).then(onCopySuccess).catch(fallbackCopy);
      } else {
        fallbackCopy();
      }

      function onCopySuccess() {
        button.textContent = 'Copied!';
        button.classList.add('copied');
        setTimeout(function() {
          button.textContent = 'Copy';
          button.classList.remove('copied');
        }, 2000);
      }

      function fallbackCopy() {
        var textarea = document.createElement('textarea');
        textarea.value = code;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand('copy');
          onCopySuccess();
        } catch (e) {
          button.textContent = 'Failed';
          setTimeout(function() { button.textContent = 'Copy'; }, 2000);
        }
        document.body.removeChild(textarea);
      }
    });
  });

  // --- 1b. Collapsible Code Blocks (> 15 lines) ---
  codeBlocks.forEach(function(block) {
    var code = block.querySelector('code');
    if (!code) return;
    var lineCount = code.textContent.split('\n').length;
    if (lineCount > 15) {
      block.classList.add('code-collapsible', 'collapsed');
      var btn = document.createElement('button');
      btn.className = 'code-expand-btn';
      btn.textContent = 'Expand (' + lineCount + ' lines)';
      block.parentNode.insertBefore(btn, block.nextSibling);
      btn.addEventListener('click', function() {
        var isCollapsed = block.classList.toggle('collapsed');
        btn.textContent = isCollapsed ? 'Expand (' + lineCount + ' lines)' : 'Collapse';
      });
    }
  });

  // --- 2. Heading Anchors ---
  var headings = document.querySelectorAll('.post-content h2, .post-content h3, .post-content h4');

  headings.forEach(function(heading) {
    if (heading.id) {
      var anchor = document.createElement('a');
      anchor.className = 'anchor-link';
      anchor.href = '#' + heading.id;
      anchor.textContent = '#';
      anchor.title = "Link to this section";
      anchor.setAttribute('aria-label', 'Link to ' + heading.textContent.trim());

      heading.appendChild(anchor);
    }
  });

  // --- 2b. Image Captions ---
  var contentArea = document.querySelector('.post-content');
  if (contentArea) {
    var captionImgs = Array.from(contentArea.querySelectorAll('p > img'));
    captionImgs.forEach(function(img) {
      if (img.closest('figure')) return;
      var alt = (img.getAttribute('alt') || '').trim();
      if (!alt) return;
      var p = img.parentElement;
      if (p.children.length !== 1) return;

      var figure = document.createElement('figure');
      figure.className = 'image-figure';
      p.parentNode.replaceChild(figure, p);
      figure.appendChild(img);

      var caption = alt.replace(/[_-]/g, ' ');
      caption = caption.charAt(0).toUpperCase() + caption.slice(1);
      var figcaption = document.createElement('figcaption');
      figcaption.className = 'image-caption';
      figcaption.textContent = caption;
      figure.appendChild(figcaption);
    });
  }

  // --- 3. Table of Contents ---
  var headingOffsets = [];
  var postContent = document.querySelector('.post-content');
  if (postContent) {
    var tocHeadings = postContent.querySelectorAll('h2[id], h3[id], h4[id]');

    // Only build TOC if post has 4+ headings (indicates a long post)
    if (tocHeadings.length >= 4) {
      // Build TOC list
      var tocHtml = '';
      tocHeadings.forEach(function(h) {
        var level = h.tagName.toLowerCase();
        var text = h.textContent.replace(/#$/, '').trim();
        tocHtml += '<li><a href="#' + h.id + '" class="toc-' + level + '">' + text + '</a></li>';
      });

      // Desktop TOC (sidebar)
      var tocWrapper = document.createElement('nav');
      tocWrapper.className = 'toc-wrapper';
      tocWrapper.setAttribute('aria-label', 'Table of contents');
      tocWrapper.innerHTML = '<div class="toc-title">On this page</div><ul class="toc-list">' + tocHtml + '</ul>';
      document.body.appendChild(tocWrapper);

      // Mobile TOC (collapsible, inserted before post content)
      var tocMobile = document.createElement('div');
      tocMobile.className = 'toc-mobile';
      tocMobile.innerHTML = '<button class="toc-mobile-toggle" aria-expanded="false">Table of Contents</button>' +
        '<div class="toc-mobile-content"><ul class="toc-list">' + tocHtml + '</ul></div>';
      postContent.insertBefore(tocMobile, postContent.firstChild);

      // Mobile toggle
      var mobileToggle = tocMobile.querySelector('.toc-mobile-toggle');
      var mobileContent = tocMobile.querySelector('.toc-mobile-content');
      mobileToggle.addEventListener('click', function() {
        var isOpen = mobileToggle.classList.toggle('open');
        mobileContent.classList.toggle('open');
        mobileToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      });

      function getHeadingOffsets() {
        var offsets = [];
        tocHeadings.forEach(function(h) {
          offsets.push({ id: h.id, top: h.getBoundingClientRect().top + window.scrollY });
        });
        return offsets;
      }

      headingOffsets = getHeadingOffsets();
      var resizeTimer;
      window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
          headingOffsets = getHeadingOffsets();
        }, 150);
      });
    }

    // --- 3b. Wrap tables for mobile scroll ---
    var tables = postContent.querySelectorAll('table');
    tables.forEach(function(table) {
      if (!table.parentElement.classList.contains('table-wrapper')) {
        var wrapper = document.createElement('div');
        wrapper.className = 'table-wrapper';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      }
    });
  }

  var scrollTicking = false;
  var tocLinks = document.querySelectorAll('.toc-wrapper .toc-list a');
  var hasToc = tocLinks.length > 0;
  window.addEventListener('scroll', function() {
    if (!scrollTicking) {
      requestAnimationFrame(function() {
        if (hasToc) {
          var scrollPos = window.scrollY + 120;
          var activeId = '';

          for (var i = headingOffsets.length - 1; i >= 0; i--) {
            if (scrollPos >= headingOffsets[i].top) {
              activeId = headingOffsets[i].id;
              break;
            }
          }

          tocLinks.forEach(function(link) {
            if (link.getAttribute('href') === '#' + activeId) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });
        }

        scrollTicking = false;
      });
      scrollTicking = true;
    }
  });

  // --- 5. Callout/Admonition Detection ---
  // Converts blockquotes starting with [!NOTE], [!TIP], [!WARNING] into styled callouts
  var blockquotes = document.querySelectorAll('.post-content blockquote');
  blockquotes.forEach(function(bq) {
    var firstP = bq.querySelector('p');
    if (!firstP) return;
    var text = firstP.innerHTML;
    var match = text.match(/^\[!(NOTE|TIP|WARNING)\]\s*/i);
    if (match) {
      var type = match[1].toLowerCase();
      bq.classList.add('callout', 'callout-' + type);
      firstP.innerHTML = text.replace(match[0], '');
    }
  });

  // --- 6. Scroll Animations ---



  // --- 7. Share Buttons (event delegation) ---
  var shareContainer = document.getElementById('share-buttons');
  if (shareContainer) {
    shareContainer.addEventListener('click', function(e) {
      var btn = e.target.closest('[data-share]');
      if (!btn) return;
      var type = btn.getAttribute('data-share');
      var url = btn.getAttribute('data-url');
      if (type === 'twitter') {
        var text = btn.getAttribute('data-text');
        var via = btn.getAttribute('data-via');
        window.open('https://twitter.com/intent/tweet?url=' + url + '&text=' + text + '&via=' + via, 'popup', 'width=600,height=600');
      } else if (type === 'linkedin') {
        window.open('https://www.linkedin.com/sharing/share-offsite/?url=' + url, 'popup', 'width=600,height=600');
      } else if (type === 'email') {
        var subject = btn.getAttribute('data-subject');
        window.location.href = 'mailto:?body=' + url + '&subject=' + subject;
      }
    });
  }

});
