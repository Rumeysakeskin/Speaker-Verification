$(document).ready(() => {
  $("#buttonCorrect").click(() => {
    $("#buttonCorrect").css('width', $("#buttonCorrect").outerWidth())
    $("#buttonCorrect").html('<i class="fa fa-spinner fa-spin"/>')
    console.log("here")
    text = $("#inputText").val()
    $.get('/api/viterbi', { text })
      .done((response) => {
        $("#buttonCorrect").html('Correggi')
        $("#correctedText").text(response)
        $("#correctedText").slideDown()
      })
  })

  $("#buttonNoise").click(() => {
    text = $("#inputText").val()
    $.get('/api/noise', { text })
      .done((response) => {
        $("#inputText").val(response)
      })
  })

  $("#inputText").autocomplete({
    source: function(request, response) {
      $.get('/api/successors', { text: request.term }, response);
    },
    minLength: 0,
    select: function( event, ui ) {
      let currentText = $("#inputText").val()
      const lastSpaceIndex = currentText.lastIndexOf(" ")
      currentText = currentText.substr(0, lastSpaceIndex)
      $("#inputText").val((currentText + " " + ui.item.label).trim() + " ")
      $('#inputText').autocomplete("search", $("#inputText").val());
      if (!$("ul.ui-autocomplete").is(":visible")) {
        $("ul.ui-autocomplete").show();
      }
      $('#inputText').trigger(jQuery.Event("keydown"));
      event.preventDefault()
    }
  });
})
