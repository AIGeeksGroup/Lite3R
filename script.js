// Theme toggle disabled - only light mode supported
// document.addEventListener('DOMContentLoaded', function() {
//     const themeBtn = document.getElementById('theme-btn');
//     const body = document.body;
//     
//     function getSystemTheme() {
//         return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
//     }
//     
//     const savedTheme = localStorage.getItem('theme');
//     const initialTheme = savedTheme || getSystemTheme();
//     
//     body.setAttribute('data-theme', initialTheme);
//     updateThemeIcon(initialTheme);
//     
//     window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
//         if (!localStorage.getItem('theme')) {
//             const newTheme = e.matches ? 'dark' : 'light';
//             body.setAttribute('data-theme', newTheme);
//             updateThemeIcon(newTheme);
//         }
//     });
//     
//     themeBtn.addEventListener('click', function() {
//         const currentTheme = body.getAttribute('data-theme');
//         const newTheme = currentTheme === 'light' ? 'dark' : 'light';
//         
//         body.setAttribute('data-theme', newTheme);
//         localStorage.setItem('theme', newTheme);
//         updateThemeIcon(newTheme);
//     });
//     
//     function updateThemeIcon(theme) {
//         const icon = themeBtn.querySelector('i');
//         icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
//     }
// });

// 应用场景轮播功能
let currentApplication = 0;
const applicationSlides = document.querySelectorAll('.application-slide');
const applicationIndicators = document.querySelectorAll('.applications .indicator');

function changeApplication(direction) {
    applicationSlides[currentApplication].classList.remove('active');
    applicationIndicators[currentApplication].classList.remove('active');
    
    currentApplication += direction;
    
    if (currentApplication >= applicationSlides.length) {
        currentApplication = 0;
    } else if (currentApplication < 0) {
        currentApplication = applicationSlides.length - 1;
    }
    
    applicationSlides[currentApplication].classList.add('active');
    applicationIndicators[currentApplication].classList.add('active');
}

function setApplication(index) {
    applicationSlides[currentApplication].classList.remove('active');
    applicationIndicators[currentApplication].classList.remove('active');
    
    currentApplication = index;
    
    applicationSlides[currentApplication].classList.add('active');
    applicationIndicators[currentApplication].classList.add('active');
}

// 技术分析轮播功能
let currentAnalysis = 0;
const analysisSlides = document.querySelectorAll('.analysis-slide');
const analysisIndicators = document.querySelectorAll('.analysis .indicator');

function changeAnalysis(direction) {
    analysisSlides[currentAnalysis].classList.remove('active');
    analysisIndicators[currentAnalysis].classList.remove('active');
    
    currentAnalysis += direction;
    
    if (currentAnalysis >= analysisSlides.length) {
        currentAnalysis = 0;
    } else if (currentAnalysis < 0) {
        currentAnalysis = analysisSlides.length - 1;
    }
    
    analysisSlides[currentAnalysis].classList.add('active');
    analysisIndicators[currentAnalysis].classList.add('active');
}

function setAnalysis(index) {
    analysisSlides[currentAnalysis].classList.remove('active');
    analysisIndicators[currentAnalysis].classList.remove('active');
    
    currentAnalysis = index;
    
    analysisSlides[currentAnalysis].classList.add('active');
    analysisIndicators[currentAnalysis].classList.add('active');
}

// Smooth scroll navigation with navbar offset
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const target = document.querySelector(targetId);
        if (target) {
            const navbarHeight = 70;
            const extraPadding = 20;
            const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navbarHeight - extraPadding;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Auto-carousel disabled
// Uncomment the code below to enable auto-carousel

// function startAutoCarousel() {
//     setInterval(() => {
//         changeApplication(1);
//     }, 5000);
// }

// let autoCarouselInterval;

// function pauseAutoCarousel() {
//     clearInterval(autoCarouselInterval);
// }

// function resumeAutoCarousel() {
//     autoCarouselInterval = setInterval(() => {
//         changeApplication(1);
//     }, 5000);
// }

// const applicationCarousel = document.querySelector('.applications-carousel');
// if (applicationCarousel) {
//     applicationCarousel.addEventListener('mouseenter', pauseAutoCarousel);
//     applicationCarousel.addEventListener('mouseleave', resumeAutoCarousel);
// }

// document.addEventListener('DOMContentLoaded', function() {
//     setTimeout(() => {
//         resumeAutoCarousel();
//     }, 3000);
// });

// Enhanced video lazy loading and streaming functionality
function initVideoStreaming() {
    // Main video streaming with progress
    const mainVideo = document.getElementById('mainVideo');
    const mainVideoLoader = document.getElementById('mainVideoLoader');
    const mainVideoProgress = document.getElementById('mainVideoProgress');
    
    if (mainVideo && mainVideoLoader) {
        // Create Intersection Observer for lazy loading
        const videoObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    loadVideoWithProgress(mainVideo, mainVideoLoader, mainVideoProgress);
                    videoObserver.unobserve(mainVideo);
                }
            });
        }, {
            rootMargin: '100px' // Start loading 100px before entering viewport
        });
        
        videoObserver.observe(mainVideo);
    }
    
    // Lazy load other videos
    lazyLoadVideos();
}

function loadVideoWithProgress(video, loader, progressBar) {
    const videoSrc = video.dataset.src;
    if (!videoSrc) return;
    
    // Show loader
    loader.classList.remove('hidden');
    
    // Create XMLHttpRequest to track loading progress
    const xhr = new XMLHttpRequest();
    xhr.open('GET', videoSrc, true);
    xhr.responseType = 'blob';
    
    // Track download progress
    xhr.onprogress = function(e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            if (progressBar) {
                progressBar.style.width = percentComplete + '%';
            }
        }
    };
    
    // On load complete
    xhr.onload = function() {
        if (xhr.status === 200) {
            const blob = xhr.response;
            const blobUrl = URL.createObjectURL(blob);
            
            // Set video source
            const source = video.querySelector('source');
            if (source) {
                source.src = blobUrl;
            }
            video.src = blobUrl;
            
            // Wait for video to be ready to play
            video.addEventListener('loadeddata', function() {
                // Hide loader
                setTimeout(() => {
                    loader.classList.add('hidden');
                    video.classList.add('loaded');
                }, 300);
                
                // Attempt autoplay
                const playPromise = video.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => {
                        // Autoplay was prevented, user needs to interact
                        console.log('Autoplay prevented:', error);
                    });
                }
            });
            
            // Load video
            video.load();
        }
    };
    
    // Error handling
    xhr.onerror = function() {
        console.error('Video loading failed');
        loader.querySelector('.loader-text').textContent = 'Loading failed, please refresh';
        if (progressBar) {
            progressBar.style.background = '#ef4444';
            progressBar.style.width = '100%';
        }
    };
    
    // Send request
    xhr.send();
}

// Standard lazy loading for other videos
function lazyLoadVideos() {
    const videos = document.querySelectorAll('video.lazy-video:not(#mainVideo)');
    
    const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                
                // Check if already loaded or loading
                if (video.dataset.isLoading === 'true' || video.dataset.isLoaded === 'true') {
                    console.log('Video already loading or loaded, skipping:', video.dataset.src);
                    return;
                }
                
                // Mark as loading
                video.dataset.isLoading = 'true';
                
                const source = video.querySelector('source[data-src]');
                
                // Find the loader element
                const loader = video.parentElement.querySelector('.video-loader');
                
                console.log('Starting to load video:', video.dataset.src);
                
                // Show loader
                if (loader) {
                    loader.classList.remove('hidden');
                }
                
                // Set up load event handler
                const onLoad = function() {
                    console.log('Video loaded successfully:', video.dataset.src);
                    video.dataset.isLoading = 'false';
                    video.dataset.isLoaded = 'true';
                    
                    setTimeout(() => {
                        if (loader) {
                            loader.classList.add('hidden');
                        }
                        video.classList.add('loaded');
                    }, 300);
                };
                
                const onError = function() {
                    console.error('Video failed to load:', video.dataset.src);
                    video.dataset.isLoading = 'false';
                    if (loader) {
                        loader.querySelector('.loader-text').textContent = 'Failed to load video';
                    }
                };
                
                video.addEventListener('loadeddata', onLoad, { once: true });
                video.addEventListener('error', onError, { once: true });
                
                // Set video sources
                if (source && source.dataset.src) {
                    source.src = source.dataset.src;
                    source.removeAttribute('data-src');
                }
                
                if (video.dataset.src) {
                    const videoSrc = video.dataset.src;
                    video.src = videoSrc;
                    video.removeAttribute('data-src');
                }
                
                video.load();
                
                // Stop observing this video
                videoObserver.unobserve(video);
            }
        });
    }, {
        rootMargin: '100px'
    });
    
    videos.forEach(video => {
        videoObserver.observe(video);
    });
}

// Initialize video streaming on page load
document.addEventListener('DOMContentLoaded', initVideoStreaming);

const sceneInputImages = {
    '01_ornate_facade': ['00000000.jpg', '00000027.jpg', '00000055.jpg', '00000082.jpg', '00000109.jpg', '00000136.jpg', '00000164.jpg', '00000191.jpg', '00000218.jpg', '00000245.jpg', '00000273.jpg', '00000300.jpg'],
    '02_curved_roof_plaza': ['00000000.jpg', '00000065.jpg', '00000130.jpg', '00000196.jpg', '00000261.jpg', '00000326.jpg', '00000391.jpg', '00000456.jpg', '00000521.jpg', '00000587.jpg', '00000652.jpg', '00000717.jpg'],
    '03_temple_complex': ['00000000.jpg', '00000053.jpg', '00000105.jpg', '00000158.jpg', '00000211.jpg', '00000263.jpg', '00000316.jpg', '00000368.jpg', '00000421.jpg', '00000474.jpg', '00000526.jpg', '00000579.jpg'],
    '04_indoor_statue': ['00000000.jpg', '00000031.jpg', '00000062.jpg', '00000094.jpg', '00000125.jpg', '00000156.jpg', '00000187.jpg', '00000218.jpg', '00000249.jpg', '00000281.jpg', '00000312.jpg', '00000343.jpg'],
    '05_relief_sculpture': ['00000000.jpg', '00000069.jpg', '00000137.jpg', '00000206.jpg', '00000274.jpg', '00000342.jpg', '00000410.jpg', '00000478.jpg', '00000547.jpg', '00000616.jpg', '00000684.jpg', '00000752.jpg'],
    '06_tabletop_sculpture': ['00000000.jpg', '00000023.jpg', '00000047.jpg', '00000070.jpg', '00000093.jpg', '00000116.jpg', '00000140.jpg', '00000163.jpg', '00000186.jpg', '00000209.jpg', '00000233.jpg', '00000256.jpg'],
    '07_excavator': ['00000000.jpg', '00000015.jpg', '00000030.jpg', '00000045.jpg', '00000060.jpg', '00000075.jpg', '00000089.jpg', '00000104.jpg', '00000119.jpg', '00000134.jpg', '00000149.jpg', '00000164.jpg'],
    '08_camera_object': ['00000000.jpg', '00000013.jpg', '00000026.jpg', '00000039.jpg', '00000052.jpg', '00000065.jpg', '00000078.jpg', '00000091.jpg', '00000104.jpg', '00000117.jpg', '00000130.jpg', '00000143.jpg']
};

const sceneViewerSlugs = {
    '01_ornate_facade': 'ornate-facade',
    '02_curved_roof_plaza': 'curved-roof-plaza',
    '03_temple_complex': 'temple-complex',
    '04_indoor_statue': 'indoor-statue',
    '05_relief_sculpture': 'relief-sculpture',
    '06_tabletop_sculpture': 'tabletop-sculpture',
    '07_excavator': 'excavator',
    '08_camera_object': 'camera-object'
};

function formatSceneName(sceneId) {
    return sceneId
        .replace(/^\d+_/, '')
        .split('_')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function updateSceneInputViews(sceneId) {
    const filenames = sceneInputImages[sceneId];
    const inputImages = document.querySelectorAll('.scene-image-strip img');
    if (!filenames || !inputImages.length) return;

    const sceneName = formatSceneName(sceneId);
    inputImages.forEach((image, index) => {
        const filename = filenames[index];
        image.src = `asset/scenes_images_8scenes/${sceneId}/images/${filename}`;
        image.alt = `${sceneName} input view ${index + 1}`;
    });

    const strip = document.querySelector('.scene-image-strip');
    if (strip) {
        strip.setAttribute('aria-label', `Input views for ${sceneName}`);
    }
}

function updateSceneViewers(sceneId) {
    const sceneSlug = sceneViewerSlugs[sceneId];
    if (!sceneSlug) return;

    document.querySelectorAll('.scene-viewer-iframe[data-viewer-kind]').forEach((iframe) => {
        const viewerKind = iframe.getAttribute('data-viewer-kind');
        const nextSrc = `html/embedded-${sceneSlug}-${viewerKind}-viewer.html`;
        if (!iframe.src.endsWith(nextSrc)) {
            iframe.src = nextSrc;
        }
    });
}

// Switch iframe content for interactive examples
function switchIframe(element) {
    console.log('switchIframe called');
    
    // Get the iframe src from data attribute
    const sceneId = element.getAttribute('data-scene');
    if (sceneId) {
        const gallery = element.closest('.thumbnail-gallery');
        const thumbnails = gallery ? gallery.querySelectorAll('.thumbnail-item') : document.querySelectorAll('.thumbnail-item');
        thumbnails.forEach(thumb => {
            thumb.classList.remove('active', 'loading');
        });
        element.classList.add('active');
        updateSceneInputViews(sceneId);
        updateSceneViewers(sceneId);
        return;
    }

    const newSrc = element.getAttribute('data-src');
    const mainIframe = document.getElementById('mainIframe');
    const iframeLoader = document.getElementById('iframeLoader');
    
    console.log('New src:', newSrc);
    console.log('Main iframe:', mainIframe);
    console.log('Loader:', iframeLoader);
    
    if (!mainIframe) {
        console.error('Main iframe not found!');
        return;
    }
    
    if (!newSrc) {
        console.error('No data-src attribute found!');
        return;
    }
    
    // Don't reload if it's the same src
    if (mainIframe.src === newSrc) {
        console.log('Same src, skipping reload');
        return;
    }
    
    console.log('Starting iframe switch...');
    
    // Show loader and hide iframe
    if (iframeLoader) {
        iframeLoader.classList.remove('hidden');
        iframeLoader.querySelector('.loader-text').textContent = 'Loading interactive demo...';
    }
    mainIframe.classList.remove('loaded');
    
    // Update active state on thumbnails
    const thumbnails = document.querySelectorAll('.thumbnail-item');
    thumbnails.forEach(thumb => {
        thumb.classList.remove('active', 'loading');
    });
    element.classList.add('active', 'loading');

    if (sceneId) {
        updateSceneInputViews(sceneId);
    }
    
    // Update iframe src
    console.log('Setting iframe src to:', newSrc);
    mainIframe.src = newSrc;
}

// Initialize iframe loading handlers
document.addEventListener('DOMContentLoaded', function() {
    const mainIframe = document.getElementById('mainIframe');
    const iframeLoader = document.getElementById('iframeLoader');
    
    if (mainIframe && iframeLoader) {
        console.log('Setting up iframe event listeners');
        
        // Handle iframe load event
        mainIframe.addEventListener('load', function() {
            console.log('Iframe loaded successfully');
            setTimeout(() => {
                iframeLoader.classList.add('hidden');
                mainIframe.classList.add('loaded');
                
                // Remove loading state from all thumbnails
                const thumbnails = document.querySelectorAll('.thumbnail-item');
                thumbnails.forEach(thumb => {
                    thumb.classList.remove('loading');
                });
            }, 500);
        });
        
        // Handle iframe error event
        mainIframe.addEventListener('error', function() {
            console.error('Iframe failed to load');
            if (iframeLoader) {
                iframeLoader.querySelector('.loader-text').textContent = 'Failed to load. Please try again.';
            }
            
            // Remove loading state from all thumbnails
            const thumbnails = document.querySelectorAll('.thumbnail-item');
            thumbnails.forEach(thumb => {
                thumb.classList.remove('loading');
            });
        });
        
        // Initial load - show iframe after it loads
        console.log('Waiting for initial iframe load...');
    } else {
        console.error('Iframe or loader not found!', { mainIframe, iframeLoader });
    }
});

// 添加页面加载动画
document.addEventListener('DOMContentLoaded', function() {
    const elements = document.querySelectorAll('.overview-card, .demo-item, .model-card');
    elements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

// 键盘导航支持
document.addEventListener('keydown', function(e) {
    // 左右箭头键控制轮播
    if (e.key === 'ArrowLeft') {
        changeApplication(-1);
        changeAnalysis(-1);
    } else if (e.key === 'ArrowRight') {
        changeApplication(1);
        changeAnalysis(1);
    }
});

// 触摸滑动支持（移动端）
let touchStartX = 0;
let touchEndX = 0;

document.addEventListener('touchstart', function(e) {
    touchStartX = e.changedTouches[0].screenX;
});

document.addEventListener('touchend', function(e) {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
});

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;
    
    if (Math.abs(diff) > swipeThreshold) {
        if (diff > 0) {
            // 向左滑动，显示下一个
            changeApplication(1);
            changeAnalysis(1);
        } else {
            // 向右滑动，显示上一个
            changeApplication(-1);
            changeAnalysis(-1);
        }
    }
}

function replayMetricBars(card) {
    card.classList.remove('bars-animated');
    card.classList.add('replay-bars');

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            card.classList.remove('replay-bars');
            card.classList.add('bars-animated');
        });
    });
}

function initMetricBarAnimations() {
    const metricCards = document.querySelectorAll('.metric-card, .result-chart-card, .ablation-chart-card');
    if (!metricCards.length) return;

    const animateCard = (card) => {
        if (!card.classList.contains('bars-animated')) {
            card.classList.add('bars-animated');
        }
    };

    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    animateCard(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.35 });

        metricCards.forEach((card) => observer.observe(card));
    } else {
        metricCards.forEach(animateCard);
    }

    metricCards.forEach((card) => {
        card.addEventListener('mouseenter', () => replayMetricBars(card));
    });
}

document.addEventListener('DOMContentLoaded', initMetricBarAnimations);

function initSceneImageZoom() {
    const sceneImages = document.querySelectorAll('.scene-image-strip img');
    if (!sceneImages.length) return;

    const preview = document.createElement('div');
    preview.className = 'scene-image-preview';
    preview.setAttribute('aria-hidden', 'true');

    const previewImage = document.createElement('img');
    previewImage.alt = '';
    preview.appendChild(previewImage);
    document.body.appendChild(preview);

    const closePreview = () => {
        preview.classList.remove('active');
        preview.setAttribute('aria-hidden', 'true');
    };

    sceneImages.forEach((image) => {
        image.addEventListener('click', () => {
            previewImage.src = image.src;
            previewImage.alt = image.alt;
            preview.classList.add('active');
            preview.setAttribute('aria-hidden', 'false');
        });
    });

    preview.addEventListener('click', closePreview);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closePreview();
        }
    });
}

document.addEventListener('DOMContentLoaded', initSceneImageZoom);
