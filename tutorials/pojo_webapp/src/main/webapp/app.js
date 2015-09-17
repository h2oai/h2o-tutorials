$(document).ready(function(){

  var $classField = $('#classField');
  var $sexField = $('#sexField');
  var $ageField = $('#ageField');
  var $fareField = $('#fareField');
  var $thumbsUpIcon = $('#thumbsUpIcon');
  var $thumbsDownIcon = $('#thumbsDownIcon');
  var $warningIcon = $('#warningIcon');

  function predict() {
    var pclass = $classField.val();
    var sex = $sexField.val();
    var age = $ageField.val();
    var fare = $fareField.val();

    var data = {
      pclass: pclass,
      sex: sex,
      age: age,
      fare: fare
    };

    var xhr = $.getJSON('/pojo_webapp/predict', data)
      .done(function(data){
        console.log(data);
        if(data.label === '0') {
          $thumbsUpIcon.hide();
          $thumbsDownIcon.show();
          $warningIcon.hide();
        } else {
          $thumbsUpIcon.show();
          $thumbsDownIcon.hide();
          $warningIcon.hide();
        }
      })
      .fail(function(error){
        console.error(error.responseText ? error.responseText : error);
        $thumbsUpIcon.hide();
        $thumbsDownIcon.hide();
        $warningIcon.show();
      })
  }

  var updatePrediction = _.debounce(predict, 250);

  $classField.change(updatePrediction);
  $sexField.change(updatePrediction);
  $ageField.keydown(updatePrediction);
  $fareField.keydown(updatePrediction);

  updatePrediction();

});

