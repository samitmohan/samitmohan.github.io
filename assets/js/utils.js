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

  // --- 3. Table of Contents ---
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

      // Active section highlighting on scroll (throttled with rAF)
      var tocLinks = tocWrapper.querySelectorAll('.toc-list a');
      var scrollTicking = false;

      function getHeadingOffsets() {
        var offsets = [];
        tocHeadings.forEach(function(h) {
          offsets.push({ id: h.id, top: h.getBoundingClientRect().top + window.scrollY });
        });
        return offsets;
      }

      var headingOffsets = getHeadingOffsets();
      window.addEventListener('resize', function() {
        headingOffsets = getHeadingOffsets();
      });

      window.addEventListener('scroll', function() {
        if (!scrollTicking) {
          requestAnimationFrame(function() {
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

            scrollTicking = false;
          });
          scrollTicking = true;
        }
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

  // --- 4. Back to Top Button ---
  var backToTop = document.createElement('button');
  backToTop.className = 'back-to-top';
  backToTop.title = 'Back to top';
  backToTop.setAttribute('aria-label', 'Scroll back to top');
  backToTop.innerHTML = '<svg viewBox="0 0 24 24"><polyline points="18 15 12 9 6 15"/></svg>';
  document.body.appendChild(backToTop);

  var backToTopTicking = false;
  window.addEventListener('scroll', function() {
    if (!backToTopTicking) {
      requestAnimationFrame(function() {
        if (window.scrollY > 500) {
          backToTop.classList.add('visible');
        } else {
          backToTop.classList.remove('visible');
        }
        backToTopTicking = false;
      });
      backToTopTicking = true;
    }
  });

  backToTop.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
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

});
