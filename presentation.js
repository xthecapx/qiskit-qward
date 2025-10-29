/**
 * qWard Presentation - Slide Navigation
 * Handles slide transitions, keyboard navigation, and counter updates
 */

document.addEventListener('DOMContentLoaded', () => {
    // Get all slide elements and navigation controls
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const slideCounter = document.getElementById('slideCounter');
    
    // Initialize slide tracking
    let currentSlide = 0;
    const totalSlides = slides.length;

    /**
     * Display a specific slide by index
     * @param {number} index - The slide index to display
     */
    function showSlide(index) {
        // Update active class for all slides
        slides.forEach((slide, i) => {
            slide.classList.toggle('active', i === index);
        });
        
        // Update slide counter display
        slideCounter.textContent = `${index + 1} / ${totalSlides}`;
        
        // Hide/show navigation buttons appropriately
        prevBtn.style.display = index === 0 ? 'none' : 'flex';
        nextBtn.style.display = index === totalSlides - 1 ? 'none' : 'flex';
    }

    /**
     * Navigate to the next slide
     */
    function nextSlide() {
        if (currentSlide < totalSlides - 1) {
            currentSlide++;
            showSlide(currentSlide);
        }
    }

    /**
     * Navigate to the previous slide
     */
    function prevSlide() {
        if (currentSlide > 0) {
            currentSlide--;
            showSlide(currentSlide);
        }
    }

    /**
     * Jump to a specific slide number (1-indexed)
     * @param {number} slideNumber - The slide number to jump to
     */
    function goToSlide(slideNumber) {
        const index = slideNumber - 1;
        if (index >= 0 && index < totalSlides) {
            currentSlide = index;
            showSlide(currentSlide);
        }
    }

    // Event Listeners
    nextBtn.addEventListener('click', nextSlide);
    prevBtn.addEventListener('click', prevSlide);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        switch(e.key) {
            case 'ArrowRight':
            case ' ': // Spacebar
            case 'PageDown':
                e.preventDefault();
                nextSlide();
                break;
            
            case 'ArrowLeft':
            case 'PageUp':
                e.preventDefault();
                prevSlide();
                break;
            
            case 'Home':
                e.preventDefault();
                goToSlide(1);
                break;
            
            case 'End':
                e.preventDefault();
                goToSlide(totalSlides);
                break;
            
            // Allow jumping to specific slides with number keys
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    const num = parseInt(e.key);
                    if (num <= totalSlides) {
                        goToSlide(num);
                    }
                }
                break;
        }
    });

    // Touch/swipe support for mobile devices
    let touchStartX = 0;
    let touchEndX = 0;

    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });

    document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });

    function handleSwipe() {
        const swipeThreshold = 50; // Minimum distance for a swipe
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                // Swipe left - next slide
                nextSlide();
            } else {
                // Swipe right - previous slide
                prevSlide();
            }
        }
    }

    // Initialize the presentation
    showSlide(currentSlide);

    // Log presentation info for debugging
    console.log(`qWard Presentation loaded: ${totalSlides} slides`);
    console.log('Navigation: Arrow keys, Space, Page Up/Down, Home, End');
    console.log('Quick jump: Ctrl/Cmd + Number (1-9)');
});

