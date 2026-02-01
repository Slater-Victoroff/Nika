// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    // Tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            this.classList.add('active');
            const targetContent = document.getElementById(tabId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });

    // Model selector functionality
    const modelButtons = document.querySelectorAll('.model-button');
    if (modelButtons.length > 0 && typeof modelData !== 'undefined') {
        modelButtons.forEach(button => {
            button.addEventListener('click', function() {
                const modelId = this.getAttribute('data-model');
                const data = modelData[modelId];
                if (!data) return;

                // Update active button
                modelButtons.forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');

                // Update video
                const video = document.getElementById('comparison-video');
                if (video) {
                    const source = video.querySelector('source');
                    if (source) {
                        source.src = data.video;
                        video.load();
                    }
                }

                // Update plot images
                const errorTimeImg = document.getElementById('error-time-img');
                const spatialImg = document.getElementById('spatial-img');
                if (errorTimeImg) errorTimeImg.src = data.errorTime;
                if (spatialImg) spatialImg.src = data.spatial;

                // Update metadata
                const psnrDisplay = document.getElementById('psnr-display');
                const metaPsnr = document.getElementById('meta-psnr');
                const metaConfig = document.getElementById('meta-config');
                const metaEpochs = document.getElementById('meta-epochs');

                if (psnrDisplay) psnrDisplay.textContent = data.psnr;
                if (metaPsnr) metaPsnr.textContent = data.psnr;
                if (metaConfig) metaConfig.textContent = data.config;
                if (metaEpochs) metaEpochs.textContent = data.epochs;
            });
        });
    }

    // Lightbox functionality
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = lightbox ? lightbox.querySelector('img') : null;
    const lightboxClose = lightbox ? lightbox.querySelector('.lightbox-close') : null;
    const lightboxTriggers = document.querySelectorAll('.lightbox-trigger');

    if (lightbox && lightboxImg) {
        // Open lightbox on image click
        lightboxTriggers.forEach(trigger => {
            trigger.addEventListener('click', function() {
                lightboxImg.src = this.src;
                lightboxImg.alt = this.alt;
                lightbox.classList.add('active');
                document.body.style.overflow = 'hidden';
            });
        });

        // Close lightbox on button click
        if (lightboxClose) {
            lightboxClose.addEventListener('click', closeLightbox);
        }

        // Close lightbox on background click
        lightbox.addEventListener('click', function(e) {
            if (e.target === lightbox) {
                closeLightbox();
            }
        });

        // Close lightbox on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && lightbox.classList.contains('active')) {
                closeLightbox();
            }
        });

        function closeLightbox() {
            lightbox.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
});
