document.getElementById('current-year').textContent = new Date().getFullYear();

(function () {
    const menuToggleBtn = document.getElementById('menuToggleBtn');
    const quickNav = document.getElementById('quickNav');

    if (!menuToggleBtn || !quickNav) {
    return;
    }

    menuToggleBtn.addEventListener('click', function () {
    const willOpen = !quickNav.classList.contains('is-open');
    quickNav.classList.toggle('is-open', willOpen);
    menuToggleBtn.setAttribute('aria-expanded', String(willOpen));
    });

    quickNav.querySelectorAll('a').forEach(function (link) {
    link.addEventListener('click', function () {
        quickNav.classList.remove('is-open');
        menuToggleBtn.setAttribute('aria-expanded', 'false');
    });
    });

    document.addEventListener('click', function (event) {
    if (!quickNav.classList.contains('is-open')) {
        return;
    }
    if (!quickNav.contains(event.target) && !menuToggleBtn.contains(event.target)) {
        quickNav.classList.remove('is-open');
        menuToggleBtn.setAttribute('aria-expanded', 'false');
    }
    });
})();

function copyCitation() {
    const citationBlock = document.getElementById('citationCode');
    const copyButton = document.getElementById('copyCitationBtn');
    const citationText = citationBlock ? citationBlock.innerText : '';

    if (!citationText) {
    return;
    }

    navigator.clipboard.writeText(citationText).then(() => {
    const originalText = copyButton.textContent;
    copyButton.textContent = 'Copied';
    setTimeout(() => {
        copyButton.textContent = originalText;
    }, 1400);
    });
}

(function () {
    const zoomableImages = document.querySelectorAll('.zoomable-image');
    const lightbox = document.getElementById('imageLightbox');
    const viewport = document.getElementById('lightboxViewport');
    const lightboxImage = document.getElementById('lightboxImage');
    const lightboxPanel = lightbox.querySelector('.lightbox-panel');
    const closeBtn = document.getElementById('lightboxCloseBtn');
    const zoomInBtn = document.getElementById('zoomInBtn');
    const zoomOutBtn = document.getElementById('zoomOutBtn');
    const zoomResetBtn = document.getElementById('zoomResetBtn');

    let scale = 1;
    const minScale = 0.5;
    const maxScale = 5;
    const zoomStep = 0.2;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let startScrollLeft = 0;
    let startScrollTop = 0;
    let isTouchDragging = false;
    let isPinching = false;
    let pinchStartDistance = 0;
    let pinchStartScale = 1;

    function getTouchDistance(touchA, touchB) {
    const deltaX = touchB.clientX - touchA.clientX;
    const deltaY = touchB.clientY - touchA.clientY;
    return Math.hypot(deltaX, deltaY);
    }

    function getTouchCenter(touchA, touchB) {
    return {
        x: (touchA.clientX + touchB.clientX) / 2,
        y: (touchA.clientY + touchB.clientY) / 2
    };
    }

    function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
    }

    function applyScale() {
    lightboxImage.style.transform = `scale(${scale})`;
    }

    function setScale(nextScale) {
    scale = clamp(nextScale, minScale, maxScale);
    applyScale();
    }

    function openLightbox(src, alt) {
    lightboxImage.onload = function () {
        const imageWidth = lightboxImage.naturalWidth;
        const imageHeight = lightboxImage.naturalHeight;

        viewport.style.width = '';
        viewport.style.height = '';
        lightboxPanel.style.width = '';

        lightbox.classList.add('is-open');
        lightbox.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';

        const availableWidth = viewport.clientWidth;
        const availableHeight = viewport.clientHeight;
        const fitScale = Math.min(availableWidth / imageWidth, availableHeight / imageHeight, 1);

        scale = clamp(fitScale, minScale, maxScale);
        applyScale();
        viewport.scrollTop = 0;
        viewport.scrollLeft = 0;
    };

    lightboxImage.src = src;
    lightboxImage.alt = alt || 'Expanded image';
    }

    function closeLightbox() {
    lightbox.classList.remove('is-open');
    lightbox.setAttribute('aria-hidden', 'true');
    lightboxImage.src = '';
    viewport.style.width = '';
    viewport.style.height = '';
    lightboxPanel.style.width = '';
    document.body.style.overflow = '';
    }

    zoomableImages.forEach((img) => {
    img.addEventListener('click', () => openLightbox(img.src, img.alt));
    });

    closeBtn.addEventListener('click', closeLightbox);
    zoomInBtn.addEventListener('click', () => setScale(scale + zoomStep));
    zoomOutBtn.addEventListener('click', () => setScale(scale - zoomStep));
    zoomResetBtn.addEventListener('click', () => setScale(1));

    lightbox.addEventListener('click', (event) => {
    if (event.target === lightbox) {
        closeLightbox();
    }
    });

    document.addEventListener('keydown', (event) => {
    if (!lightbox.classList.contains('is-open')) {
        return;
    }
    if (event.key === 'Escape') {
        closeLightbox();
    } else if (event.key === '+' || event.key === '=') {
        setScale(scale + zoomStep);
    } else if (event.key === '-') {
        setScale(scale - zoomStep);
    } else if (event.key === '0') {
        setScale(1);
    }
    });

    viewport.addEventListener('wheel', (event) => {
    if (!lightbox.classList.contains('is-open')) {
        return;
    }
    event.preventDefault();
    const direction = event.deltaY > 0 ? -zoomStep : zoomStep;
    setScale(scale + direction);
    }, { passive: false });

    viewport.addEventListener('mousedown', (event) => {
    if (!lightbox.classList.contains('is-open')) {
        return;
    }
    isDragging = true;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    startScrollLeft = viewport.scrollLeft;
    startScrollTop = viewport.scrollTop;
    });

    viewport.addEventListener('touchstart', (event) => {
    if (!lightbox.classList.contains('is-open')) {
        return;
    }

    if (event.touches.length === 2) {
        isPinching = true;
        isTouchDragging = false;
        pinchStartDistance = getTouchDistance(event.touches[0], event.touches[1]);
        pinchStartScale = scale;
        return;
    }

    if (event.touches.length === 1) {
        isPinching = false;
        isTouchDragging = true;
        dragStartX = event.touches[0].clientX;
        dragStartY = event.touches[0].clientY;
        startScrollLeft = viewport.scrollLeft;
        startScrollTop = viewport.scrollTop;
    }
    }, { passive: true });

    viewport.addEventListener('touchmove', (event) => {
    if (isPinching && event.touches.length === 2) {
        event.preventDefault();

        const currentDistance = getTouchDistance(event.touches[0], event.touches[1]);
        if (!pinchStartDistance) {
        return;
        }

        const pinchRatio = currentDistance / pinchStartDistance;
        const center = getTouchCenter(event.touches[0], event.touches[1]);
        const viewportRect = viewport.getBoundingClientRect();
        const focalX = center.x - viewportRect.left;
        const focalY = center.y - viewportRect.top;

        const imageX = viewport.scrollLeft + focalX;
        const imageY = viewport.scrollTop + focalY;
        const previousScale = scale;

        setScale(pinchStartScale * pinchRatio);

        const scaleRatio = scale / previousScale;
        viewport.scrollLeft = imageX * scaleRatio - focalX;
        viewport.scrollTop = imageY * scaleRatio - focalY;
        return;
    }

    if (isTouchDragging && event.touches.length === 1) {
        event.preventDefault();
        viewport.scrollLeft = startScrollLeft - (event.touches[0].clientX - dragStartX);
        viewport.scrollTop = startScrollTop - (event.touches[0].clientY - dragStartY);
    }
    }, { passive: false });

    viewport.addEventListener('touchend', (event) => {
    if (event.touches.length === 0) {
        isTouchDragging = false;
        isPinching = false;
        return;
    }

    if (event.touches.length === 1) {
        isPinching = false;
        isTouchDragging = true;
        dragStartX = event.touches[0].clientX;
        dragStartY = event.touches[0].clientY;
        startScrollLeft = viewport.scrollLeft;
        startScrollTop = viewport.scrollTop;
    }
    });

    viewport.addEventListener('touchcancel', () => {
    isTouchDragging = false;
    isPinching = false;
    });

    window.addEventListener('mousemove', (event) => {
    if (!isDragging) {
        return;
    }
    viewport.scrollLeft = startScrollLeft - (event.clientX - dragStartX);
    viewport.scrollTop = startScrollTop - (event.clientY - dragStartY);
    });

    window.addEventListener('mouseup', () => {
    isDragging = false;
    });
})();