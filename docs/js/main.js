function checkSubmit(e) {
   if(e && e.keyCode == 13) {
      document.forms[0].submit();
   }
}
$(function() {
 $('#ctrySearch').submit(function(){
   var ctry = $('#searchInput').val();
   $(this).attr('action', ctry + ".html");
 });
});
