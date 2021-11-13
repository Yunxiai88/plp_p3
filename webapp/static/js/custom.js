$(document).ready(function () {
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

    $('#uploadForm').on('submit', function(event) {
        event.preventDefault();

        var fd = new FormData();
        var files = $('#sfile')[0].files;

        if(files.length > 0 ){
            $('.loader').show()
            fd.append('file',files[0])

            // disable button
            $('#closeBtn').prop('disabled', true)
            $('#submitBtn').prop('disabled', true)

            $.ajax({
                url: $(this).attr('action'),
                type: 'post',
                data: fd,
                contentType: false,
                processData: false,
                success: function(response) {
                    if(response != 0) {
                        $('.loader').hide()
                        $('#uploadModal').modal('hide');
                    }
                }
            }).fail(function(response) {
                $('.loader').hide()
                $('#uploadModal').modal('hide');
            });
        } else {
            alert("Please select a file.")
        }
    });

    //image pop up
    $(function() {
		$('.pop').on('click', function() {
			$('.imagepreview').attr('src', $(this).find('img').attr('src'));
			$('#imagemodal').modal('show');   
		});		
    });

});

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