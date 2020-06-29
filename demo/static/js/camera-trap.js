var carousel = $("#carousel").waterwheelCarousel({
    flankingItems: 2,
    sizeMultiplier:0.9,
    opacityMultiplier: 0.4,
    startingItem: 1, // item to place in the center of the carousel. Set to 0 for auto
    separation: 120, // distance between items in carousel
    
});
$('#prev').bind('click', function () {
    carousel.prev();
    return false
});
$('#next').bind('click', function () {
    carousel.next();
    return false;
});
$('#reload').bind('click', function () {
    newOptions = eval("(" + $('#newoptions').val() + ")");
    carousel.reload(newOptions);
    return false;
});

// Noty.overrideDefaults({
//     layout   : 'topRight',
//     theme    : 'mint',
//     closeWith: ['click', 'button'],
//     timeout: 1500,
//     animation: {
//         open : 'animated fadeInRight',
//         close: 'animated fadeOutRight'
//     }
// });


$('#export_result').on('click', function(){
    var result_det = $(this).val()
    result_det = JSON.parse(result_det)

    rows = [['Image Name', 'Bboxes', 'Status', 'Number of objects']]

    image_id = Object.keys(result_det)
    for(let i=0; i< image_id.length; i++){
        rows.push([image_id[0], result_det[image_id[i]]['bboxes'], result_det[image_id[i]]['message'], result_det[image_id[i]]['num_objects']])
    }
    let csvContent = "data:text/csv;charset=utf-8,";
    rows.forEach(function(rowArray){
        let row = rowArray.join(",");
        csvContent += row + "\r\n";
    });
    var encodedUri = encodeURI(csvContent);
    window.open(encodedUri); 
})

function myFunc(vars) {
    console.log(vars)
    return vars
}

function back_uplaod(){
    console.log($('#upload-url').val())
    location.href = '/'
}

function reset_upload(){
    location.href = '/upload'
}

function onSubmit(){
    $(".loading").css("visibility", "visible")
    $("#overlay").show();
    $.ajax({
        'url': '/processurlimage?image_url=' + $('#upload-url').val(),
        'method': 'GET'
    }).then(function(response){
        $(".loading").css("visibility", "hidden")
        $("#overlay").hide();
        document.write(response)
    }, function(error){
        $(".loading").css("visibility", "hidden")
        $("#overlay").hide();
        alert("Invalid URL")
    })
}

$('img[rel=popover]').popover({
  html: true,
  trigger: 'hover',
  placement: 'right',
  content: function(){return '<img src="'+$(this).data('img') + 
  '" style="width:100%;height:100%;" />';}
});

function FileUploadErrors(error)
{
    console.log(error)
    if(error != "Your can't upload any more files.")
    {
        new Noty({ 
            type: 'error', 
            text: error,
        }).show();
    }  
    else{
        maxFilesExceededError.show();
    }              
}

(function() {
    // Initialize
    //var bLazy = new Blazy({
    //    container: '.scroll-class'
    //});

    maxFilesExceededError = new Noty({ 
        type:'error', 
        text: 'Sorry, only 10 files allowed. Extra files have been removed',
        //timeout: 2500
    });

    if(localStorage.getItem("error"))
    {
        new Noty({ 
            type:'error', 
            text: localStorage.getItem("error"),
            timeout: 2500
        }).show();
        localStorage.clear();
    }

    $('.carousel-image').on('click', function(){
        click_id = $(this).attr('id').replace('carousel-', 'table-')
        $('.table-details').removeClass('row-background')
        $('#' + click_id).addClass('row-background')
    
    })

})();  
