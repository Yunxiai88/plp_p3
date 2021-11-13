$(document).ready(function () {
    var date;
    var log= {}
    $.getJSON( "data/summary.json", function( data ) {
        log = data;
    });

    $('#verticalScroll').DataTable({
        "scrollY": "50vh",
        "scrollCollapse": true,
        "order": [[ 0, "desc" ]]
    });
    $('.dataTables_length').addClass('bs-select');

    //graph selection
    $(".dropdown-topic a").click(function(){
        var selText = $(this).text();
        $("#topic-text").text(selText);

        var img = "/data/img/"+selText+".jpg";
        $("#img-a").attr("src", img);
    });

    //date selection
    $(".drapdown-sentiment a").click(function(){
        var selText = $(this).text();
        $("#sentiment-text").text(selText);

        var img = "/data/img/"+selText+".jpg";
        $("#img-b").attr("src", img);
    });

    $('#process').on('click', getPolarity);

    function getPolarity() {
        var review = $("#review").val();
    
        $.ajax({
            url: '/process',
            data: {
                'review': review
            },
            type: 'POST',
            success: function (data) {
                var content = '';
                $.each(data, function (i, item) {
                    content += item
                });
                $("#result").html(content)
            }
        });
    }

    //image pop up
    $(function() {
		$('.pop').on('click', function() {
			$('.imagepreview').attr('src', $(this).find('img').attr('src'));
			$('#imagemodal').modal('show');   
		});		
    });

});