// --- 0. Theme Toggle (runs immediately, before DOMContentLoaded) ---
(function() {
  var saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.body.classList.add('dark-mode');
  }
})();

document.addEventListener("DOMContentLoaded", function() {

  // Theme toggle button
  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function() {
      document.body.classList.toggle('dark-mode');
      var isDark = document.body.classList.contains('dark-mode');
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
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

      navigator.clipboard.writeText(code).then(function() {
        button.textContent = 'Copied!';
        button.classList.add('copied');

        setTimeout(function() {
          button.textContent = 'Copy';
          button.classList.remove('copied');
        }, 2000);
      }).catch(function(err) {
        console.error('Failed to copy: ', err);
      });
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
      tocWrapper.innerHTML = '<div class="toc-title">On this page</div><ul class="toc-list">' + tocHtml + '</ul>';
      document.body.appendChild(tocWrapper);

      // Mobile TOC (collapsible, inserted before post content)
      var tocMobile = document.createElement('div');
      tocMobile.className = 'toc-mobile';
      tocMobile.innerHTML = '<button class="toc-mobile-toggle">Table of Contents</button>' +
        '<div class="toc-mobile-content"><ul class="toc-list">' + tocHtml + '</ul></div>';
      postContent.insertBefore(tocMobile, postContent.firstChild);

      // Mobile toggle
      var mobileToggle = tocMobile.querySelector('.toc-mobile-toggle');
      var mobileContent = tocMobile.querySelector('.toc-mobile-content');
      mobileToggle.addEventListener('click', function() {
        mobileToggle.classList.toggle('open');
        mobileContent.classList.toggle('open');
      });

      // Active section highlighting on scroll
      var tocLinks = tocWrapper.querySelectorAll('.toc-list a');
      var headingOffsets = [];

      function updateHeadingOffsets() {
        headingOffsets = [];
        tocHeadings.forEach(function(h) {
          headingOffsets.push({ id: h.id, top: h.offsetTop });
        });
      }

      updateHeadingOffsets();
      window.addEventListener('resize', updateHeadingOffsets);

      window.addEventListener('scroll', function() {
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
      });
    }
  }

  // --- 4. Reading Progress Bar (removed) ---

  // --- 5. Back to Top Button ---
  var backToTop = document.createElement('button');
  backToTop.className = 'back-to-top';
  backToTop.title = 'Back to top';
  backToTop.innerHTML = '<svg viewBox="0 0 24 24"><polyline points="18 15 12 9 6 15"/></svg>';
  document.body.appendChild(backToTop);

  window.addEventListener('scroll', function() {
    if (window.scrollY > 500) {
      backToTop.classList.add('visible');
    } else {
      backToTop.classList.remove('visible');
    }
  });

  backToTop.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // --- 6. Callout/Admonition Detection ---
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

  // --- 7. Project Filter Buttons ---
  var filterBar = document.querySelector('.project-filter-bar');
  if (filterBar) {
    var filterBtns = filterBar.querySelectorAll('.filter-btn');
    var projectCards = document.querySelectorAll('.project-card');

    filterBtns.forEach(function(btn) {
      btn.addEventListener('click', function() {
        var filter = btn.getAttribute('data-filter');

        // Toggle active state
        filterBtns.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');

        projectCards.forEach(function(card) {
          if (filter === 'all') {
            card.classList.remove('hidden');
          } else {
            var tag = card.getAttribute('data-tag');
            if (tag === filter) {
              card.classList.remove('hidden');
            } else {
              card.classList.add('hidden');
            }
          }
        });
      });
    });
  }
});
