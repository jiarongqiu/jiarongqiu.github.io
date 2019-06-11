!function() {
    var v = document.getElementById("cross-portfolio")   
    classie.add(v, "modify");
    
    $(".sub-title").typed({
        strings: ["Web Developer", "Web Designer"],
        typeSpeed: 1,
        backSpeed: 1,
        backDelay: 1400,
        loop: !0
    }), 
    new Waypoint({
        element: document.getElementById("count"),
        handler: function() {
            $(".count").countTo()
        },
        offset: 500
    });
    smoothScroll.init({
        speed: 1e3
    }), $(".carousel-inner").owlCarousel({
        navigation: !1,
        slideSpeed: 300,
        paginationSpeed: 400,
        singleItem: !0,
        autoPlay: 3e3
    }), window.sr = ScrollReveal().reveal(".animated"), $(".gmap").mobileGmap({
        markers: [{
            position: "center",
            info: "121 S Pinckney St",
            showInfo: !0
        }]
    })
}();

